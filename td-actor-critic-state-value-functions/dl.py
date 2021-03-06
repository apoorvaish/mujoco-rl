import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, lr, input_dims, n_hidden=64, output_dims=4):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dims, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.pi = nn.Linear(n_hidden, output_dims)
        self.v = nn.Linear(n_hidden, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # self.device = (torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        # self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v_s = self.v(x)
        return pi, v_s
    