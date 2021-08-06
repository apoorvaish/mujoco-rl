# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

# %%
class Network(nn.Module):
    def __init__(self, n_states, n_actions, lr, gamma):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.fc1 = nn.Linear(n_states, 128)
        # self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(128, n_actions)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.to(self.device)
    
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)
        x = f.sigmoid(self.fc1(x))
        # x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def learn(self, state, action, reward, state_):
        self.optimizer.zero_grad()
        q_state = self.forward(state)[action]
        q_state_ = self.forward(state_).max()     
        q_target = reward + (self.gamma *  q_state_)
        # print(q_state, "q' ", q_state_, "Target", target_q)
        loss = self.loss(q_target, q_state)
        loss.backward()
        self.optimizer.step()
        
        
# %%
