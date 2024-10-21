import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np 
from tqdm import tqdm
from utils import save_test_1, save_test_2
from sksurv.metrics import concordance_index_ipcw, brier_score
from metric import BS, DDC, DCAL, CI

class Model(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.threshold = kwargs['threshold']
        self.input_dim = kwargs['input_dim']
        self.contrastive_network = kwargs['contrastive_network'].to(self.args.device)
        self.survival_network = kwargs['survival_network'].to(self.args.device)
        self.optimizer_contrastive = kwargs['optimizer_contrastive']
        self.optimizer_survival = kwargs['optimizer_survival']
        self.corruption_rate = kwargs['corruption_rate']
        self.num_category = kwargs['num_category']
        self.corruption_len = int(self.corruption_rate * self.input_dim)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.eval_times = kwargs['eval_times']

    def info_nce_loss(self, features, time, label):
        # make mask matrix
        mask_matrix = torch.cat([torch.arange(self.args.batch_size) for i in range(2)], dim=0)
        mask_matrix = (mask_matrix.unsqueeze(0) == mask_matrix.unsqueeze(1)).float()
        
        # make similarity matrix 
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both : mask_matrix and similarity_matrix
        mask = torch.eye(mask_matrix.shape[0], dtype=torch.bool)
        mask_matrix = mask_matrix[~mask].view(mask_matrix.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        
        # make time matrix
        time = time.reshape(-1,1).repeat(2,1)
        ones_time = torch.ones_like(time, dtype=torch.float)
        time_matrix_A = torch.matmul(time, ones_time.transpose(0,1))
        time_matrix_B = torch.matmul(ones_time, time.transpose(0,1))
        time_matrix = time_matrix_A - time_matrix_B

        # make label matrix
        label = label.reshape(-1,1).repeat(2,1)
        ones_label = torch.ones_like(label)
        label_matrix_A = torch.matmul(label, ones_label.transpose(0,1))
        label_matrix_A[label_matrix_A==1] = 2 
        label_matrix_B = torch.matmul(ones_label, label.transpose(0,1))
        label_matrix = label_matrix_A - label_matrix_B

        incomparable_pair_1 = (label_matrix==2) & (time_matrix > 0)
        incomparable_pair_2 = (label_matrix==-1) & (time_matrix < 0)
        
        comparable_pair_1 = (label_matrix==2) & (time_matrix <= 0)
        comparable_pair_2 = (label_matrix==-1) & (time_matrix >= 0)

        label_matrix[incomparable_pair_1] = 0
        label_matrix[incomparable_pair_2] = 0

        label_matrix[comparable_pair_1] = 3
        label_matrix[comparable_pair_2] = 3

        # weight_matrix
        weight_matrix= (1. - torch.exp(-torch.cdist(time.reshape([-1,1]), time.reshape([-1,1]), p=1)/self.args.sigma))

        # margin setting
        not_assure_margin_idx = (label_matrix==3) & (weight_matrix <= self.threshold)
        weight_matrix[not_assure_margin_idx] = 0.0 

        # [cen, cen]
        weight_matrix[np.where(label_matrix == 0)] = 0.0 

        # discard main_diag from weight_matrix
        weight_matrix = weight_matrix[~mask].view(weight_matrix.shape[0],-1).to(self.args.device)

        # select and combine positives
        positives = similarity_matrix[mask_matrix.bool()].view(mask_matrix.shape[0], -1)

        # select only the negatives 
        negatives = similarity_matrix[~mask_matrix.bool()].view(similarity_matrix.shape[0], -1)
        negatives_weight = weight_matrix[~mask_matrix.bool()].view(similarity_matrix.shape[0],-1) 

        if self.args.version == 'prod':

            df_positive=pd.DataFrame(positives.detach().cpu())
            df_negative=pd.DataFrame(negatives.detach().cpu().flatten())

            negatives_weight = negatives_weight * self.reciprocala(negatives_weight.mean(dim=-1,keepdim=True)) 
            negatives = negatives * negatives_weight

            logits = torch.cat([positives, negatives], dim=1)
            logits = logits / self.args.temperature 

            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        elif self.args.version == 'out':

            positive_logit = positives
            negative_logit = negatives

            positive_logit = torch.exp(positive_logit/self.args.temperature)
            negative_logit = torch.exp(negative_logit/self.args.temperature)

            negatives_weight = negatives_weight * self.reciprocala(negatives_weight.mean(dim=-1,keepdim=True)) # normalization
            negative_logit = negatives_weight * negative_logit
       
            negative_logit = negative_logit.sum(dim=-1)
            positive_logit = positive_logit.squeeze(1)

            logits = (positive_logit / (positive_logit + negative_logit))
            logits = -1 * self.log(logits)

            logits = logits.mean()
            labels = None
            
        return logits,labels

    def train(self, train_loader, train_loader_for_eval, valid_loader):
        
        for epoch_counter in range(self.args.epochs):

            self.contrastive_network.train()
            self.survival_network.train()

            total_loss_1 = 0
            total_loss_2 = 0
            
            for anchor, positive, time, label, mask_1, mask_2 in train_loader:  
                batch_size, m = anchor.size()
                corruption_mask = torch.zeros_like(anchor, dtype=torch.bool) 

                for i in range(batch_size):
                    corruption_idx = torch.randperm(m)[: self.corruption_len]
                    corruption_mask[i, corruption_idx] = True

                positive = torch.where(corruption_mask, positive, anchor)
                
                data = torch.cat([anchor, positive], dim=0).to(self.args.device)

                # Contrastive_Network (Loss_1)
                projection_head_output = self.contrastive_network(data)
                if  self.args.version=='prod':
                    weighted_softmax_logits_log,labels  = self.info_nce_loss(projection_head_output, time, label)
                    loss_1 = self.criterion(weighted_softmax_logits_log, labels)  

                elif self.args.version =='out':
                    weighted_softmax_logits_log, _ = self.info_nce_loss(projection_head_output, time, label)
                    loss_1 = weighted_softmax_logits_log

                self.optimizer_contrastive.zero_grad()
                loss_1.backward()
                self.optimizer_contrastive.step()

                # Survival_Network (Loss_2)
                mask_1 = mask_1.to(self.args.device)
                mask_2 = mask_2.to(self.args.device)
                label = label.to(self.args.device)

                anchor_expand = (torch.unsqueeze(anchor,-1).expand(batch_size, m, self.num_category)).transpose(1,2).reshape(-1,m).to(self.args.device)
                hazard = self.survival_network(anchor_expand, 'train')
                hazard = hazard.reshape(batch_size, self.num_category)               
                uncensor = self.uncensor(hazard, mask_1, mask_2, label) 
                censor = self.censor(hazard, mask_2, label)
                loss_2 = uncensor + censor  #Loss_2
                
                self.optimizer_survival.zero_grad()
                loss_2.backward()
                self.optimizer_survival.step()

                total_loss_1 += loss_1.item()*self.args.batch_size
                total_loss_2 += loss_2.item()*self.args.batch_size
                           
            self.contrastive_network.eval()
            self.survival_network.eval()
            with torch.no_grad():
                for train_data, _, time, label, _, _ in train_loader_for_eval:
                    X_tr = train_data
                    T_tr = time
                    E_tr = label
                
                for valid_data, _, time, label, _, _ in valid_loader:  
                    X_val = valid_data.to(self.args.device)
                    T_val = time
                    E_val = label

                    X_val_expand = torch.unsqueeze(X_val,-1).expand(X_val.shape[0], X_val.shape[1], self.num_category).transpose(1,2).reshape(-1,X_val.shape[1]).to(self.args.device)
                    out = self.survival_network(X_val_expand, 'valid').reshape(X_val.shape[0], self.num_category).cpu()
                    H = out.cumsum(1)
                    S = torch.exp(-H)
                    W = 1-S
                
                tr_y_structured =  [(np.asarray(E_tr)[i], np.asarray(T_tr)[i]) for i in range(len(E_tr))]
                tr_y_structured = np.array(tr_y_structured, dtype=[('status', 'bool'),('time','<f8')])

                val_y_structured =  [(np.asarray(E_val)[i], np.asarray(T_val)[i]) for i in range(len(E_val))]
                val_y_structured = np.array(val_y_structured, dtype=[('status', 'bool'),('time','<f8')])    

                # to compute brier-score without error
                T_tr_ = pd.DataFrame(T_tr, columns=['time'])
                T_val_ = pd.DataFrame(T_val, columns=['time'])
                T_val_.loc[T_val_['time'] > T_tr_['time'].max(), 'time'] = T_tr_['time'].max()

                val_y_structured_ =  [(np.asarray(E_val)[i], np.asarray(T_val_)[i]) for i in range(len(E_val))]
                val_y_structured_ = np.array(val_y_structured_, dtype=[('status', 'bool'),('time','<f8')])

                RESULTS1 = np.zeros([len(self.eval_times)])
                RESULTS2 = np.zeros([len(self.eval_times)])

                for t, eval_time in enumerate(self.eval_times):
                    RESULTS1[t] = concordance_index_ipcw(tr_y_structured, val_y_structured, W[:, eval_time], tau=eval_time)[0]
                    RESULTS2[t] = brier_score(tr_y_structured, val_y_structured_, S[:, eval_time], times=eval_time)[1][0]
                
                time_dependent_cindex = np.round(RESULTS1, 4)
                time_dependent_brier_score = np.round(RESULTS2, 4)

                # DCAL
                dcal = DCAL()
                predicted_proba = np.array([])

                for i in range(np.shape(T_val_)[0]):
                    predicted_proba = np.append(predicted_proba, S[i, int(np.array(T_val.squeeze())[i])])
                p_value, _ = dcal.d_calibration(predicted_proba, np.array(E_val.squeeze()), 10)

                # CI
                Concordance_Index = CI()
                S, T_tr, E_tr, T_val, E_val = np.array(S), np.array(T_tr), np.array(E_tr), np.array(T_val), np.array(E_val)
                cindex, _, _= Concordance_Index.concordance_cal(S, T_val, E_val)
                                                                
                #DDC
                ddc = DDC()
                binned = ddc.binned_dist(S, T_val, E_val, 10)

                if len(binned) == 10:
                    ddc_val = ddc.ddc_cal(np.array(binned)/sum(binned),np.ones(10)*.1)

                # BS
                time_coordinates = np.arange(S.shape[1])
                S, T_tr, E_tr, T_val, E_val = np.array(S), np.array(T_tr), np.array(E_tr), np.array(T_val), np.array(E_val)
                bs = BS(S, time_coordinates, T_tr, E_tr, T_val, E_val)
                bs = bs.integrated_brier_score(num_points=None, IPCW_weighted=True, draw_figure=True)

                if epoch_counter == 0:
                    top_score = time_dependent_cindex.mean()

                elif (epoch_counter > 0) and (time_dependent_cindex.mean() > top_score): 
                    top_score = time_dependent_cindex.mean()
                    
                    if not os.path.exists('./{}/'.format(self.args.data_name)): os.makedirs('./{}/'.format(self.args.data_name))
                    torch.save(self.survival_network, './{}/NLL_SNCE_{}.pt'.format(self.args.data_name, self.args.seed))

    
            print("Epoch: {} | loss_1: {:.4f} | loss_2: {:.4f} | CI_td: {:.4f} | BS_td: {:.4f} | CI: {:.4f} | BS: {:.4f} | DDC: {:.4f} | DCAL: {:.4f}".format(
                epoch_counter, total_loss_1/len(train_loader.dataset), total_loss_2/len(train_loader.dataset), 
                time_dependent_cindex.mean(), time_dependent_brier_score.mean(), 
                cindex, bs, ddc_val, p_value
            ))

    
    def test(self, train_data_for_test, test_loader):

        self.survival_network = torch.load('./{}/NLL_SNCE_{}.pt'.format(self.args.data_name, self.args.seed))
        
        self.survival_network.eval()
        with torch.no_grad():
            for train_data, _, time, label, _, _ in train_data_for_test:
                X_tr = train_data
                T_tr = time
                E_tr = label
            
            for test_data, _, time, label, _, _ in test_loader:  
                X_te = test_data.to(self.args.device)
                T_te = time
                E_te = label

                X_te_expand = torch.unsqueeze(X_te,-1).expand(X_te.shape[0], X_te.shape[1], self.num_category).transpose(1,2).reshape(-1,X_te.shape[1]).to(self.args.device)
                out = self.survival_network(X_te_expand, 'test').reshape(X_te.shape[0], self.num_category).cpu()
                H = out.cumsum(1)
                S = torch.exp(-H)
                W = 1-S
            
            tr_y_structured =  [(np.asarray(E_tr)[i], np.asarray(T_tr)[i]) for i in range(len(E_tr))]
            tr_y_structured = np.array(tr_y_structured, dtype=[('status', 'bool'),('time','<f8')])

            te_y_structured =  [(np.asarray(E_te)[i], np.asarray(T_te)[i]) for i in range(len(E_te))]
            te_y_structured = np.array(te_y_structured, dtype=[('status', 'bool'),('time','<f8')])    

            # to compute brier-score without error
            T_te_ = pd.DataFrame(T_te, columns=['time'])
            T_tr_ = pd.DataFrame(T_tr, columns=['time'])
            T_te_.loc[T_te_['time'] > T_tr_['time'].max(), 'time'] = T_tr_['time'].max()

            te_y_structured_ =  [(np.asarray(E_te)[i], np.asarray(T_te_)[i]) for i in range(len(E_te))]
            te_y_structured_ = np.array(te_y_structured_, dtype=[('status', 'bool'),('time','<f8')])

            time_dependent_cindex = []
            time_dependent_brier_score = []
            for t, eval_time in enumerate(self.eval_times):
                temp1 = concordance_index_ipcw(tr_y_structured, te_y_structured, W[:, eval_time], tau=eval_time)[0]
                temp2= brier_score(tr_y_structured, te_y_structured_, S[:, eval_time], times=eval_time)[1][0]

                time_dependent_cindex.append(temp1)
                time_dependent_brier_score.append(temp2)

            # DCAL
            dcal = DCAL()
            predicted_proba = np.array([])

            for i in range(np.shape(T_te_)[0]):
                predicted_proba = np.append(predicted_proba, S[i, int(np.array(T_te.squeeze())[i])])
            p_value, bin_statistics = dcal.d_calibration(predicted_proba, np.array(E_te.squeeze()), 10)

            # CI
            Concordance_Index = CI()
            S, T_tr, E_tr, T_te, E_te = np.array(S), np.array(T_tr), np.array(E_tr), np.array(T_te), np.array(E_te)
            cindex, _, _= Concordance_Index.concordance_cal(S, T_te, E_te)
                                                             
            #DDC
            ddc = DDC()
            binned = ddc.binned_dist(S, T_te, E_te, 10)
            
            if len(binned) == 10:
                ddc_val = ddc.ddc_cal(np.array(binned)/sum(binned),np.ones(10)*.1)
            elif len(binned) == 11:
                ddc_val = ddc.ddc_cal(np.array(binned)/sum(binned),np.ones(11)*.1)

            # BS
            time_coordinates = np.arange(S.shape[1])
            S, T_tr, E_tr, T_te, E_te = np.array(S), np.array(T_tr), np.array(E_tr), np.array(T_te), np.array(E_te)
            bs = BS(S, time_coordinates, T_tr, E_tr, T_te, E_te)
            bs = bs.integrated_brier_score(num_points=None, IPCW_weighted=True, draw_figure=True)

            performace_data_1 = {
                'SEED' : [self.args.seed],
                'CI' : [cindex],
                'BS' : [bs],
                'DDC' : [ddc_val],
                'DCAL' : [p_value],
            }

            save_test_1(self.args.data_name, performace_data_1)
            save_test_2(self.args.data_name, time_dependent_cindex, time_dependent_brier_score, self.args.seed)


    def uncensor(self, h, mask_1, mask_2, label):
        # h(t)
        p1 = self.log(h) * mask_1
        p1 = p1.sum(dim=-1)

        # (1-h(t))
        p2 = self.log(1-h) * mask_2
        p2 = p2.sum(dim=-1)

        # neg log likelihood
        p = p1 + p2

        # indicator
        p = p * label.squeeze()
        p = -(p.mean(dim=-1))

        return p
    
    def censor(self, h, mask2, label):

        # (1-h(t))
        p = self.log(1-h) * mask2
        p = p.sum(dim=-1)
        
        p = p * (1-(label.squeeze()))
        p = -(p.mean(dim=-1))

        return p
    
    def log(self, x):
        return torch.log(x + 1e-8)

    def reciprocala(self, x): 
        return torch.where(x != 0, torch.reciprocal(x), torch.tensor(0.0))

