#%%
import gym
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing
from multiprocessing.dummy import Pool
from dl import Network
from es import evolution_strategy


# Unsolved Environments
# Humanoid-V2
# HumanoidStandup-V2
# Walker2d-v2
# Reacher-V2
# Ant-V2
# environment
ENV_NAME = 'Swimmer-v2'
env = gym.make(ENV_NAME)


# thread pool for parallelization
pool = Pool(4)

## hyperparameters
D = len(env.reset())            # Input layer : No. of states = 17
M = 100                         # Hidden Layer
K = env.action_space.shape[0]   # Output Layer : No. of actions = 6
action_max = env.action_space.high[0]
env.close()

def reward_function(params):
    model = Network(D, M, K, action_max)
    model.set_params(params)
    # play one episode and return the total reward
    env = gym.make(ENV_NAME)
    episode_reward = 0
    episode_length = 0 # not sure if it will be used
    done = False
    state = env.reset()
    while not done:
        # get the action
        action = model.sample_action(state)
        # perform the action
        state, reward, done, _ = env.step(action)
        # update total reward
        episode_reward += reward
        episode_length += 1
    env.close()
    return episode_reward

def init_model(D, M, K, num_iters=100):
    # train and save
    model = Network(D, M, K, action_max)
    model.init()
    rewards = []
    params = model.get_params()
    for i in range(num_iters):
        best_params, offspring_rewards, time = evolution_strategy(func=reward_function,
                                                    population_size=30,
                                                    sigma=0.1,
                                                    lr=0.05,
                                                    initial_params=params,
                                                    num_iters=num_iters,
                                                    pool = pool)
        # plot the rewards per iteration
        mu = offspring_rewards.mean()
        print("Iter:", i, "Avg Reward: %.3f" % mu, "Max:", offspring_rewards.max(), "Duration:", time)
        rewards.append(mu)
        plt.plot(rewards)
        plt.pause(0.05)
        
        plt.savefig('./model-results/' + ENV_NAME + '_training.png')
        model.set_params(best_params)
        np.savez('./model-results/' + ENV_NAME + '-es-mujoco-results.npz' ,
                    train=rewards,
                    **model.get_params_dict())
    # plt.show()     
          
if __name__ == '__main__':
    init_model(D, M, K, num_iters=100)
        
# %%
# 