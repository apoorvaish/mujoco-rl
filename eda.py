# %%
# https://github.com/openai/gym/blob/master/gym/envs/mujoco/swimmer_v3.py
# https://github.com/openai/gym/issues/585
import gym
env = gym.make('Swimmer-v2')

action_space = env.action_space.shape[0]
observation_space = env.observation_space
action_max = env.action_space.high[0]

done = False
state = env.reset()

while not done:
    print(state)
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    env.render()
# %%
