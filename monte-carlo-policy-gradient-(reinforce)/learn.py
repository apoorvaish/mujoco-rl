# %%
import gym
import numpy as np
from rl import Agent
from save_runs import GifWriter

class Trainer:
    def __init__(self, env, gif_freq=100):
        self.init_environment(env)
        self.init_agent()
        self.gif_freq = gif_freq
        if gif_freq:         
            self.gif_writer = GifWriter()         
        else:
            self.gif_writer = False   
    
    def init_environment(self, env):
        self.env = gym.make(env)
        self.obs_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0] # .n for lunarlander, .shape[0] for mujoco
        self.init_agent()
        
    def init_agent(self):
        self.agent = Agent(obs_space=self.obs_space,
                        action_space=self.action_space,
                        lr=5e-3, gamma=0.99)
        
        self.gif_writer = GifWriter()
        
    def play_ep(self, render=False):
        obs = self.env.reset()
        done = False
        return_ = 0
        frames = []
        # Start the episode
        while not done:
            action = self.agent.choose_action(obs)
            obs_, reward, done, info = self.env.step(action)
            self.agent.store_reward(reward)
            obs = obs_
            return_ += reward
            if render:
                frame = self.env.render(mode='rgb_array')
                frames.append(frame)         
        return return_, frames
        
    
    def train(self, n_eps):                    
        returns  = []
        for ep in range(n_eps):
            if self.gif_freq:
                if ep % self.gif_freq == 0:
                    return_, frames = self.play_ep(render=True)
                    self.gif_writer.save_gif(
                        frames, f'./{self.env.unwrapped.spec.id}-{ep}')
                else:
                    return_, _ = self.play_ep(render=False)
            else:
                return_, _ = self.play_ep(render=False)
            # Update NN parameters 
            self.agent.learn()
            # For user
            returns.append(return_)
            print(f'Episode: {ep} ------ Return: {return_} ------ Avg. Return: {np.mean(returns[-100:])}')
                 
                
    def test(self, render=True):
        return_, frames = self.play_ep(render=render)
        if render:     
            if self.gif_writer == False:
                self.gif_writer = GifWriter()
            self.gif_writer.save_gif(
                frames, f'./{self.env.unwrapped.spec.id}-test')
        print(f'--------------- Test Return: {return_} ---------------') 

# %%
trainer = Trainer(env='Swimmer-v2', gif_freq=0)
trainer.train(n_eps=3000)
# trainer.test()

# %%
trainer.test()

# %%
