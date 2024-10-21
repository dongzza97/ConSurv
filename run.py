import argparse
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np


from torch.utils.data import DataLoader
from learning import *
from dataset import *
from model import *

parser = argparse.ArgumentParser(description='Rank')
parser.add_argument('--seed', default=111, type=int)
parser.add_argument('--data', metavar='DIR', default='./datasets')
parser.add_argument('--data-name',type=str, default='METABRIC')
parser.add_argument('--disable-cuda', action='store_true')
parser.add_argument('--workers', default=12, type=int)
parser.add_argument('--epochs', default=2000, type=int)
parser.add_argument('--version', type=str, default='out')
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--hidden-dim', default=16, type=int)
parser.add_argument('--depth', default=4, type=int)
parser.add_argument('--drop-out', default=0.2, type=float)
parser.add_argument('--corruption-rate', default=0.5, type=float)
parser.add_argument('--quantile', default=10.0, type=float)
parser.add_argument('--lr1', default=0.0001, type=float)
parser.add_argument('--lr2', default=0.0001, type=float)
parser.add_argument('--temperature', default=0.07, type=float)
parser.add_argument('--sigma', default=0.75, type=float)
parser.add_argument('--eval_times', default=[64, 128, 192], type=int, nargs='+')
parser.add_argument('--gpu-index', default=2, type=int)

def main():
  args = parser.parse_args()

  # seed
  torch.cuda.empty_cache()
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  # check if gpu training is available
  if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    cudnn.deterministic = True
    cudnn.benchmark = True
  else: args.device = torch.device('cpu')

  dataset = Datasets(args.data, args.data_name, args.seed)

  ## train ##
  (train_data, train_times, train_label, train_mask_1,  train_mask_2),(_),(_) = dataset.get_dataset(mode='train')
  ## validation ##
  (train_data_for_eval, train_times_for_eval, train_label_for_eval, train_mask_1_for_eval,  train_mask_2_for_eval),\
  (valid_data, valid_times, valid_label, valid_mask_1,  valid_mask_2),\
  (_) = dataset.get_dataset(mode='valid')
  ## Test ##
  (train_data_for_test, train_times_for_test, train_label_for_test, train_mask_1_for_test, train_mask_2_for_test),\
  (_),\
  (test_data, test_times, test_label, test_mask_1,  test_mask_2) = dataset.get_dataset(mode='test')

  # Dataset
  train_dataset = CustomDataset(train_data, train_times, train_label, train_mask_1, train_mask_2)
  train_dataset_for_eval = CustomDataset(train_data_for_eval, train_times_for_eval, train_label_for_eval, train_mask_1_for_eval, train_mask_2_for_eval)
  train_dataset_for_test = CustomDataset(train_data_for_test, train_times_for_test, train_label_for_test, train_mask_1_for_test, train_mask_2_for_test)
  valid_dataset = CustomDataset(valid_data, valid_times, valid_label, valid_mask_1, valid_mask_2)
  test_dataset = CustomDataset(test_data, test_times, test_label, test_mask_1,  test_mask_2)

  # DataLoader
  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
  train_loader_for_eval = DataLoader(train_dataset_for_eval, batch_size=train_dataset_for_eval.shape[0], shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
  train_loader_for_test = DataLoader(train_dataset_for_test, batch_size=train_dataset_for_test.shape[0], shuffle=False, num_workers=12, pin_memory=True, drop_last=True)
  valid_loader = DataLoader(valid_dataset, batch_size=valid_dataset.shape[0], shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
  test_loader = DataLoader(test_dataset, batch_size=test_dataset.shape[0], shuffle=False, num_workers=12, pin_memory=True, drop_last=True)
  
  # Variable
  num_category = dataset.num_category
  input_dim =  dataset.data_dim[1]
  hidden_dim = args.hidden_dim 
  drop_out = args.drop_out    
  depth = args.depth
  sigma = args.sigma
  corruption_rate = args.corruption_rate
  quantile = args.quantile
  device = args.device
  batch_size = args.batch_size
  eval_times = args.eval_times
  batch_size_eval = dataset.valid_data_dim[0]
  batch_size_test = dataset.test_data_dim[0]

  # set threshold
  threshold = dataset.get_threshold(train_times, train_label, quantile, sigma) 
  
  # modeling
  encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, depth=depth, drop_out=drop_out)
  contrastive_network = ProjectionHead(encoder=encoder, hidden_dim=hidden_dim, depth=depth)
  survival_network = HazardNetwork(encoder=encoder, hidden_dim=hidden_dim+1, 
                                    num_category=num_category, 
                                    batch_size = batch_size, batch_size_eval= batch_size_eval, batch_size_test = batch_size_test, device = device
                                    )
  
  # optimizer 
  optimizer_contrastive = torch.optim.Adam(contrastive_network.parameters(), args.lr1)  
  optimizer_survival = torch.optim.Adam(survival_network.parameters(), args.lr2)
  
  # print("============================= Train =============================")
  with torch.cuda.device(args.gpu_index):
      model = Model(contrastive_network=contrastive_network, survival_network=survival_network, 
                      optimizer_contrastive=optimizer_contrastive, optimizer_survival=optimizer_survival, 
                      threshold=threshold, input_dim=input_dim, num_category=num_category, corruption_rate=corruption_rate, eval_times=eval_times, args=args) 
      model.train(train_loader, train_loader_for_eval, valid_loader)


  # print("============================= Test & Inference =============================")
      model.test(train_loader_for_test, test_loader)


if __name__ == "__main__":
  main()



