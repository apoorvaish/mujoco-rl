import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, lr, input_dims, n_hidden=100, output_dims=4):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_dims, n_hidden)
        # self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, output_dims)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # self.device = (torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        # self.to(self.device)
    
    def forward(self, obs):
        obs_tensor = torch.tensor(obs.astype(np.float32))
        x = F.relu(self.fc1(obs_tensor))
        # x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        # x = self.fc3(x)
        return x
    
