# %%
import gym
import numpy as np
from rl import Agent

# For live plots
from animate_plot import plot_learning_curve

env = gym.make('Swimmer-v2')
n_actions = env.action_space.shape[0]
n_states = env.observation_space.shape[0]

n_eps = 30000
rewards = []
batch = 100
plot_window = 10
avg_rewards = []
epsilon_hist = []
agent = Agent(n_actions=n_actions, n_states=n_states, n_eps=n_eps)

for episode in range(n_eps):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.choose_action(state)
        state_, reward, done, info = env.step(action)
        agent.network.learn(state, action, reward, state_)
        total_reward += reward
        state = state_
        agent.update_eps()
        rewards.append(total_reward)    
        if (episode % batch == 0):
            env.render()
            avg_rewards.append(np.mean(rewards[-batch:]))
            epsilon_hist.append(agent.eps)
            if (len(avg_rewards) % plot_window == 0):
                x = [(i+1)*(batch) for i in range(len(avg_rewards))]
                plot_learning_curve(x, avg_rewards, epsilon_hist)

# %%

# %%
