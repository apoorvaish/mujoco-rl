# %%
import gym
import numpy as np
from dl import Network

# %%
class Agent():
    def __init__(self, n_actions, n_states, n_eps, 
                 lr=0.0001, gamma=0.99, eps_max=1.0, eps_min = 0.01):
        self.n_actions = n_actions
        self.network = Network(n_states = n_states, n_actions= n_actions, lr = lr, gamma = gamma)
        self.eps = eps_max
        self.eps_min = eps_min
        self.eps_dec = (eps_max-eps_min) / 2500
    
    def choose_action(self, state):
        random = np.random.uniform(0,1)
        if random > self.eps:
            # Greedy action
            return self.network.forward(state).argmax().item()
        else:
            # Random action
            return np.random.choice(range(self.n_actions))
     
    def update_eps(self):
        self.eps -= self.eps_dec
        self.eps = max(self.eps, self.eps_min)
        return True
# %%
