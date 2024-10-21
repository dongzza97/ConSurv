import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class Datasets:
    def __init__(self, root_folder, data_name, seed):
        self.root_folder = root_folder
        self.data_name = data_name
        self.seed = seed
        
    def get_dataset(self, mode=None):
        data = pd.read_csv(os.path.join(self.root_folder,f'{self.data_name}.csv'), index_col=0)
        
        time = data[['time']] if self.data_name == 'GBSG' else data[['time']]//30.0
        label = data[['label']]
        data = data.iloc[:,:-2]
        num_category = int(np.max(time) * (1.02 if self.data_name in ['SEER', 'SUPPORT'] else 1.01))
        
        mask_1 = self.f_get_fc_mask2(time, label, num_category)
        mask_2 = self.f_get_fc_mask3(time, label, num_category)

        (train_data, test_data, train_times, test_times, train_label, test_label,
        train_mask_1, test_mask_1, train_mask_2, test_mask_2) = \
        train_test_split(data, time, label, mask_1, mask_2, test_size=0.2, random_state=self.seed)

        (train_data, valid_data, train_times, valid_times, train_label, valid_label,
        train_mask_1, valid_mask_1, train_mask_2, valid_mask_2) = \
        train_test_split(train_data, train_times, train_label, train_mask_1, train_mask_2, test_size=0.2, random_state=self.seed)

        constant_cols = [c for c in data.columns if data[c].nunique()==1]
        train_data.drop(columns=constant_cols, inplace=True)
        valid_data.drop(columns=constant_cols, inplace=True)
        test_data.drop(columns=constant_cols, inplace=True)

        scaler = MinMaxScaler()
        if mode == 'train':     
            train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
            valid_data = pd.DataFrame(scaler.transform(valid_data), columns=valid_data.columns)
            # train_times = pd.DataFrame(scaler.fit_transform(train_times), columns=train_times.columns)
            # valid_times = pd.DataFrame(scaler.transform(valid_times), columns=valid_times.columns)
        elif mode == 'valid':
            train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
            valid_data = pd.DataFrame(scaler.transform(valid_data), columns=valid_data.columns)
        elif mode == 'test':
            train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
            test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)
        
        # access
        self.num_category = num_category
        self.data_dim = data.shape
        self.valid_data_dim = valid_data.shape
        self.test_data_dim = test_data.shape
        
        return (train_data, train_times, train_label, train_mask_1, train_mask_2), \
               (valid_data, valid_times, valid_label, valid_mask_1, valid_mask_2), \
               (test_data, test_times, test_label, test_mask_1, test_mask_2)
                

    
    def f_get_fc_mask2(self, time, label, num_Category):
        time = np.asarray(time)
        label = np.asarray(label)

        mask = np.zeros([np.shape(time)[0], num_Category])
        for i in range(np.shape(time)[0]):
            if label[i,0] != 0:  #not censored
                mask[i, int(time[i,0])] = 1
            else: #label[i,2]==0: censored
                mask[i, int(time[i,0]+1):] =  1 #fill 1 until from the censoring time
        return mask
    
    def f_get_fc_mask3(self, time, label, num_Category):
        time = np.asarray(time)
        label = np.asarray(label)

        mask = np.zeros([np.shape(time)[0], num_Category])
        for i in range(time.shape[0]):
            time_idx = int(time[i])
            if label[i] != 0:                   # uncensor
                if time_idx != 0:               # first time pass (do nothing)
                    mask[i, :time_idx] = 1      # before event time = 1
            else:                               # censor
                mask[i, :time_idx+1] = 1        # until censor time = 1
        return mask
    
    def get_threshold(self, train_times, train_label, quantile, sigma):

        time = torch.tensor(train_times.values).reshape(-1,1).repeat(2,1)
        ones_time = torch.ones_like(time, dtype=torch.double)
        time_matrix_A = torch.matmul(time, ones_time.transpose(0,1))
        time_matrix_B = torch.matmul(ones_time, time.transpose(0,1))
        time_matrix = time_matrix_A - time_matrix_B        
        
        label = torch.tensor(train_label.values).reshape(-1,1).repeat(2,1)
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

        weight_matrix= (1. - torch.exp(-torch.cdist(time.reshape([-1,1]), time.reshape([-1,1]), p=1)/sigma))
        not_assure_margin_idx = (label_matrix==3)
        weight_matrix = weight_matrix[not_assure_margin_idx]
        threshold = np.percentile(weight_matrix, quantile, method='nearest').item()

        return threshold
    
class CustomDataset(Dataset):
    def __init__(self, data, times, label, mask_1, mask_2):
        self.data = np.array(data)
        self.times = np.array(times)
        self.label = np.array(label)
        self.mask_1 = np.array(mask_1)
        self.mask_2 = np.array(mask_2)

    def __getitem__(self, index):
        #random index for augmentation
        random_idx = np.random.randint(0, len(self))

        sample = torch.tensor(self.data[index], dtype=torch.float)
        random_sample = torch.tensor(self.data[random_idx], dtype=torch.float)
        times = torch.tensor(self.times[index], dtype=torch.float)
        label = torch.tensor(self.label[index],dtype=torch.float)
        mask_1 = torch.tensor(self.mask_1[index],dtype=torch.float)
        mask_2 = torch.tensor(self.mask_2[index], dtype=torch.float)

        return sample, random_sample, times, label, mask_1, mask_2

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape