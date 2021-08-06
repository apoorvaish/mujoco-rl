import torch
import torch.nn.functional as F
from dl import Network
import numpy as np


class Agent():
    def __init__(self, lr, obs_space, action_space, gamma=0.99):
        self.gamma = gamma
        self.lr = lr

        self.policy = Network(input_dims=obs_space, lr=self.lr, 
                                output_dims=action_space)
            
    def choose_action(self, obs):
        obs_tensor = torch.tensor(obs.astype(np.float32))
        probabilities = F.softmax(self.policy.forward(obs_tensor))
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.action_memory.append(log_probs)
        return action.item()
    
    def store_reward(self, reward):
        self.reward_memory.append(reward)
    
    def learn(self, state, reward, state_, done):
        self.actor_critic.optimizer.zero_grad()
        _, v_state = self.actor_critic.forward(state)
        _, v_state_ = self.actor_critic.forward(state_)
        # Updating the actor network
        # (1-done) give 0 value for state_ for last step
        delta = reward + self.gamma * v_state_ * (1 - int(done)) - v_state
        actor_loss = -self.log_prob * delta
        critic_loss = delta**2
        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()
        
