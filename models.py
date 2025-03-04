import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F


import math
from datetime import datetime

from torch.optim.lr_scheduler import ExponentialLR
# from ignite.handlers import create_lr_scheduler_with_warmup

class DynamicModel(nn.Module):
    def __init__(self, T, I, a = 1, K = 10, H_u = 5, H_v = 5, layers = 1, layers_v = 1, nodes = 64):
        """
        Deep Dynamic NMF with Z information
        T : time period
        I : number of categorys/items
        a : parameter to for time smoothing 
        K : chosen number of topics
        H_u : number of features in z_user information 
        H_v : number of features in z_item information (not used for now)
        layers : number of hidden layers (not used for now)
        layers_v : number of hidden layers for V matrix (not used for now)
        nodes : number of nodes in layers
        """
        super(DynamicModel, self).__init__()
        self.hidden_size = K

        self.register_buffer("last", torch.tensor([1]))

        self.K = K
        self.T = T
        self.a = a

        self.v_i = nn.parameter.Parameter(torch.zeros((K, I)))
        self.fc_u_z = nn.Linear(H_u, K)
        nn.init.kaiming_uniform_(self.fc_u_z.weight)
        nn.init.kaiming_uniform_(self.v_i)

        self.fc_ins = nn.ModuleList([nn.Linear(I + H_u, nodes) for t in range(T)])
        for fc_in in self.fc_ins:
            nn.init.kaiming_uniform_(fc_in.weight)
        self.i2o = nn.Linear(nodes, K)
        self.l2o = nn.Linear(K, K)
        nn.init.kaiming_uniform_(self.i2o.weight)
        nn.init.kaiming_uniform_(self.l2o.weight)

        self.from_last = nn.Linear(K, K)
        self.fc_v_z = nn.Linear(H_v, K)
        nn.init.kaiming_uniform_(self.fc_v_z.weight)
        self.input_bn = nn.BatchNorm1d(I)
        self.input_bn_v = nn.BatchNorm1d(H_v)
        self.bns = nn.ModuleList([nn.BatchNorm1d(I+H_u) for i in range(T)])
    def update_state(self, data, last, a = 1):
        # input = torch.cat((data, last_hidden), 1)
        # data = F.relu(self.fc_in(data))
        from_in = self.i2o(data)
        from_last = self.l2o(last)
        # output = F.softmax(from_in + from_last, dim = 1)
        # output = F.softmax(from_in, dim = 1)
        
        output = a * F.relu(from_in + from_last) + (1-a) * from_last
        
        return output

    def forward(self, x_i, z_u, z_v, input_t):
        last = torch.zeros((x_i.size(0) , self.K), device=self.i2o.weight.device)
        output_list = []
        for t in range(input_t):
            x_ = x_i[:, t]
            x_ = torch.concat((x_, z_u), dim = 1)
            x_ = self.bns[t](x_)
            x_ = F.relu(self.fc_ins[t](x_))
            last = self.update_state(x_, last, self.a)
            out = (last)
            output_list.append(out)

        v_i_ = F.relu(self.v_i)
        return output_list, v_i_

class DynamicModelNZ(nn.Module):
    def __init__(self, T, I, a = 1, K = 10, n_nodes = 64, layers = 1, layers_v = 1, nodes = 64):
        
        super(DynamicModelNZ, self).__init__()
        self.hidden_size = K

        self.register_buffer("last", torch.tensor([1]))

        self.K = K
        self.T = T
        self.a = a
        self.v_i = nn.parameter.Parameter(torch.zeros((K, I)))
        nn.init.kaiming_uniform_(self.v_i)
        self.fc_ins = nn.ModuleList([nn.Linear(I, nodes) for t in range(T)])
        for fc_in in self.fc_ins:
            nn.init.kaiming_uniform_(fc_in.weight)
        self.i2o = nn.Linear(nodes, K)
        self.l2o = nn.Linear(K, K)
        nn.init.kaiming_uniform_(self.i2o.weight)
        nn.init.kaiming_uniform_(self.l2o.weight)
        self.bns = nn.ModuleList([nn.BatchNorm1d(I) for i in range(T)])
    def update_state(self, data, last, a = 1):
        # input = torch.cat((data, last_hidden), 1)
        # data = F.relu(self.fc_in(data))
        from_in = self.i2o(data)
        from_last = self.l2o(last)
        # output = F.softmax(from_in + from_last, dim = 1)
        # output = F.softmax(from_in, dim = 1)
        
        output = a * F.relu(from_in + from_last) + (1-a) * from_last
        
        return output

    def forward(self, x_i, z_u, z_v, input_t):

        # common_uk_ = self.fc_u_z(z_u)
        common_uk_ = 0
        last = torch.zeros((x_i.size(0) , self.K), device=self.i2o.weight.device)
        output_list = []
        for t in range(input_t):
            x_ = x_i[:, t]
            x_ = self.bns[t](x_)
            x_ = F.relu(self.fc_ins[t](x_))
            last = self.update_state(x_, last, self.a)
            out = last
            output_list.append(out)
        v_i_ = F.relu(self.v_i)
        return output_list, v_i_


class DynamicModelNoT(nn.Module):
    def __init__(self, T, I, a = 1, K = 10, H_u = 5, H_v = 5, layers = 1, layers_v = 1, nodes = 64):
        
        super(DynamicModelNoT, self).__init__()
        self.hidden_size = K

        self.register_buffer("last", torch.tensor([1]))

        self.K = K
        self.T = T
        self.a = a
        self.v_i = nn.parameter.Parameter(torch.zeros((K, I)))
        self.fc_u_z = nn.Linear(H_u, K)
        nn.init.kaiming_uniform_(self.fc_u_z.weight)
        nn.init.kaiming_uniform_(self.v_i)
        self.fc_ins = nn.ModuleList([nn.Linear(I + H_u, nodes) for t in range(T)])
        for fc_in in self.fc_ins:
            nn.init.kaiming_uniform_(fc_in.weight)
        self.i2o = nn.Linear(nodes, K)
        self.l2o = nn.Linear(K, K)
        nn.init.kaiming_uniform_(self.i2o.weight)
        nn.init.kaiming_uniform_(self.l2o.weight)
        self.from_last = nn.Linear(K, K)
        self.fc_v_z = nn.Linear(H_v, K)
        nn.init.kaiming_uniform_(self.fc_v_z.weight)
        self.input_bn = nn.BatchNorm1d(I)
        self.input_bn_v = nn.BatchNorm1d(H_v)
        self.bns = nn.ModuleList([nn.BatchNorm1d(I+H_u) for i in range(T)])
    def update_state(self, data, last, a = 1):
        # input = torch.cat((data, last_hidden), 1)
        # data = F.relu(self.fc_in(data))
        from_in = self.i2o(data)
        from_last = self.l2o(last)
        # output = F.softmax(from_in + from_last, dim = 1)
        # output = F.softmax(from_in, dim = 1)
        
        output = a * F.relu(from_in) + (1-a) * from_last
        
        return output

    def forward(self, x_i, z_u, z_v, input_t):
        last = torch.zeros((x_i.size(0) , self.K), device=self.i2o.weight.device)
        output_list = []
        for t in range(input_t):
            x_ = x_i[:, t]
            x_ = torch.concat((x_, z_u), dim = 1)
            x_ = self.bns[t](x_)
            x_ = F.relu(self.fc_ins[t](x_))
            last = self.update_state(x_, last, self.a)
            out = (last)
            output_list.append(out)

        v_i_ = F.relu(self.v_i)
        return output_list, v_i_


class DynamicModelNZNT(nn.Module):
    def __init__(self, T, I, a = 1, K = 10, n_nodes = 64, layers = 1, layers_v = 1, nodes = 64):
        super(DynamicModelNZNT, self).__init__()
        self.hidden_size = K
        self.register_buffer("last", torch.tensor([1]))

        self.K = K
        self.T = T
        self.a = a
        self.v_i = nn.parameter.Parameter(torch.zeros((K, I)))
        nn.init.kaiming_uniform_(self.v_i)
        self.fc_ins = nn.ModuleList([nn.Linear(I, nodes) for t in range(T)])
        for fc_in in self.fc_ins:
            nn.init.kaiming_uniform_(fc_in.weight)
        self.i2o = nn.Linear(nodes, K)
        self.l2o = nn.Linear(K, K)
        nn.init.kaiming_uniform_(self.i2o.weight)
        nn.init.kaiming_uniform_(self.l2o.weight)
        self.bns = nn.ModuleList([nn.BatchNorm1d(I) for i in range(T)])
    def update_state(self, data, last, a = 1):
        # input = torch.cat((data, last_hidden), 1)
        # data = F.relu(self.fc_in(data))
        from_in = self.i2o(data)
        from_last = self.l2o(last)
        # output = F.softmax(from_in + from_last, dim = 1)
        # output = F.softmax(from_in, dim = 1)
        
        output = a * F.relu(from_in) + (1-a) * from_last
        
        return output

    def forward(self, x_i, z_u, z_v, input_t):

        # common_uk_ = self.fc_u_z(z_u)
        common_uk_ = 0
        last = torch.zeros((x_i.size(0) , self.K), device=self.i2o.weight.device)
        output_list = []
        for t in range(input_t):
            x_ = x_i[:, t]
            x_ = self.bns[t](x_)
            x_ = F.relu(self.fc_ins[t](x_))
            last = self.update_state(x_, last, self.a)
            out = last
            output_list.append(out)
        v_i_ = F.relu(self.v_i)
        return output_list, v_i_





class CustomDataset(Dataset):
    """
    to create pytorch dataset
    dimension : (T, batch, features)
    """
    def __init__(self, x_tensor, y_tensor, z_user = None):

        self.xdomain = x_tensor
        self.ydomain = y_tensor
        self.z_user = z_user
        
    def __getitem__(self, index):
        if self.z_user is not None:
            return (self.xdomain[:, index, :], self.ydomain[:, index, :], self.z_user[index, :])
        else:
            return (self.xdomain[:, index, :], self.ydomain[:, index, :], 0)
    
    def __len__(self):
        return self.xdomain.size(1)