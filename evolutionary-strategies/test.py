#%%
import gym
import sys
import imageio
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import multiprocessing
from multiprocessing.dummy import Pool
from dl import Network
from es import evolution_strategy
from mujoco_py.mjviewer import save_video
import cv2
import datetime
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

def play_episode(params, display=True, out=True):
    model = Network(D, M, K, action_max)
    model.set_params(params)
    # play one episode and return the total reward
    env = gym.make(ENV_NAME)
    episode_reward = 0
    done = False
    state = env.reset()
    while not done:
        # display the env
        if display:
            if out==False:     
                continue
            else:
                frame = env.render(mode='rgb_array')
                out.write(frame)                 
        # get the action
        action = model.sample_action(state)
        # perform the action
        state, reward, done, _ = env.step(action)
        print(state[3])
        # update total reward
        episode_reward += reward
    return episode_reward
def load_model(D, M, K, 
               num_iters=300, 
               train=False,
               out=False):
    # train and save
    model = Network(D, M, K, action_max)
    model.init()
    j = np.load('./model-results/' + ENV_NAME + '-es-mujoco-results.npz')
    print('Model loaded ..')
    best_params = np.concatenate([j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])
    # in case initial shapes are not correct
    D, M = j['W1'].shape
    K = len(j['b2'])
    model.D, model.M, model.K = D, M, K
    model.set_params(best_params)
    play_episode(model.get_params(), display=True, out=out)    
  
if __name__ == '__main__':
    time = datetime.datetime.now().strftime("%H-%M-%S-%z-%B-%d-%Y")
    save_dir = './model-results/video/' + ENV_NAME + '-' + time + '-output.avi'
    print(save_dir)
    out = cv2.VideoWriter(save_dir, cv2.VideoWriter_fourcc('M','J','P','G'), 60, (500, 500))
    out = load_model(D, M, K, num_iters=10, train=False, out=out)
    # out.release()

# %%
