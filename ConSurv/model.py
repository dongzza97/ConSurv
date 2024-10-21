import torch
import torch.nn as nn
import numpy as np 
import torchvision.models as models
import pdb


class Encoder(nn.Module):

    def __init__(self, 
                 input_dim,
                 hidden_dim, 
                 depth, 
                 drop_out
                 ):
        super(Encoder,self).__init__()
        
        self.encoder = MutilLayer(input_dim, hidden_dim, depth, drop_out)

    def forward(self, x):
        representation = self.encoder(x)
        return representation

class ProjectionHead(nn.Module):
    def __init__(self,
                 encoder,
                 hidden_dim,
                 depth,
                ):
        super(ProjectionHead, self).__init__()
    
        self.encoder = encoder
        self.projection_head = MutilLayer2(hidden_dim, hidden_dim, depth)

    def forward(self, x):
        representation = self.encoder(x)
        out = self.projection_head(representation)
        return out
    
class HazardNetwork(nn.Module):
    def __init__(self, 
                 encoder, hidden_dim,
                 num_category, 
                 batch_size, batch_size_eval, batch_size_test, device):
        super(HazardNetwork, self).__init__()
        
        self.encoder = encoder
        self.hazard_network = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.num_category = num_category
        self.batch_size = batch_size
        self.batch_size_eval = batch_size_eval
        self.batch_size_test = batch_size_test
        self.device = device 
    def forward(self, x, mode):
        representation = self.encoder(x)

        if mode == 'train':
            times = ((self.min_max_normalization(torch.tensor(np.arange(0, self.num_category)))).unsqueeze(0).expand(self.batch_size, 1, self.num_category).reshape(-1,1)).to(self.device)
        
        elif mode == 'valid':
            times = ((self.min_max_normalization(torch.tensor(np.arange(0, self.num_category)))).unsqueeze(0).expand(self.batch_size_eval, 1, self.num_category).reshape(-1,1)).to(self.device)
        
        elif mode == 'test':
            times = ((self.min_max_normalization(torch.tensor(np.arange(0, self.num_category)))).unsqueeze(0).expand(self.batch_size_test, 1, self.num_category).reshape(-1,1)).to(self.device)

        representation = torch.concat((representation, times),dim=1)
        hazard = self.hazard_network(representation)
        return hazard
    
    def min_max_normalization(self, tensor):
        min_value = torch.min(tensor)
        max_value = torch.max(tensor)

        normalized_tensor = (tensor - min_value) / (max_value - min_value)

        return normalized_tensor


class MutilLayer(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)

class MutilLayer2(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, n_layers):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)

