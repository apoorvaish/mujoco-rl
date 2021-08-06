import torch
import torch.nn.functional as F
from dl import Network
import numpy as np


class Agent():
    def __init__(self, lr, obs_space, action_space, gamma=0.99):
        self.gamma = gamma
        self.lr = lr
        # Memory and CPU
        self.reward_memory = []
        self.action_memory = []
        self.policy = Network(input_dims=obs_space, lr=self.lr, 
                                output_dims=action_space)
    def reset_memory(self):    
        self.action_memory = []
        self.reward_memory = []
            
    def choose_action(self, obs):        
        action_tensor = self.policy.forward(obs)
        low, high = -1.0, 1.0
        action = action_tensor.detach().numpy()
        action = action + np.random.uniform(low, high, size=action_tensor.shape) * 0.1
        action = np.clip(action, low, high)
        self.action_memory.append(torch.log(action_tensor))
        return action
    
    def store_reward(self, reward):
        self.reward_memory.append(reward)
    
    def learn(self):
        self.policy.optimizer.zero_grad()
        returns = np.zeros_like(self.reward_memory, dtype=np.float64)
        # Replay episode
        for step in range(len(self.reward_memory)):
            step_return = 0
            discount = 1
            # For each step count the future disconted reward
            for k in range(step, len(self.reward_memory)):
                step_return += self.reward_memory[k] * discount
                discount *= self.gamma
            returns[step] = step_return
        # Convert to tensor
        returns = torch.tensor(returns, dtype=torch.float)
        loss = 0
        for step_return, logprob in zip(returns, self.action_memory):
            loss += torch.sum(-step_return * logprob)
        loss.backward()
        self.policy.optimizer.step()
        self.reset_memory()
        
