import cv2
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))
import warnings
warnings.filterwarnings('ignore')

import gym
import gym_cartpole_world

from models.dqn_gen import DoubleDQNAgent


SCREEN_X = 64
SCREEN_Y = 64

MAX_FRAMES = 40
MAX_TRIALS = 10000

cartpole_version = sys.argv[1]

env_name = ('CartPoleWorld-' + cartpole_version).strip()
env = gym.make(env_name)
env.initialize()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

theta_threshold = env.theta_threshold
x_threshold = env.x_threshold

if cartpole_version[1] == '0':
    dataset_name = './dataset/Cartpole/' + 'G_len{}_xthreshold{}_thetathreshold{}_trial{}'.format(
        MAX_FRAMES, x_threshold, theta_threshold, MAX_TRIALS)
elif cartpole_version[1] == '1':
    dataset_name = './dataset/Cartpole/' + 'M_len{}_xthreshold{}_thetathreshold{}_trial{}'.format(
        MAX_FRAMES, x_threshold, theta_threshold, MAX_TRIALS)
elif cartpole_version[1] == '2':
    dataset_name = './dataset/Cartpole/' + 'N_len{}_xthreshold{}_thetathreshold{}_trial{}'.format(
        MAX_FRAMES, x_threshold, theta_threshold, MAX_TRIALS)

if not os.path.exists(dataset_name):
    os.makedirs(dataset_name)

if not os.path.exists(os.path.join(dataset_name, cartpole_version)):
    os.makedirs(os.path.join(dataset_name, cartpole_version))

agent = DoubleDQNAgent(state_size, action_size)

env = gym.wrappers.Monitor(env, "save_demo", video_callable=False, force=True, write_upon_reset=False)

trial = 0
while trial < MAX_TRIALS:
    recording_state = []
    recording_obs = []
    recording_action = []
    recording_reward = []

    success = 0

    state = env.reset()

    for frame in range(MAX_FRAMES):
        # add state
        recording_state.append(state)

        # add obs
        obs = env.render(mode='rgb_array')
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (128, 128), interpolation=cv2.INTER_CUBIC)
        obs = cv2.normalize(obs, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        recording_obs.append(obs)

        # add action
        state = np.reshape(state, [1, state_size])
        action = 1 if np.random.rand() > 0.5 else 0
        recording_action.append(action)

        # add reward
        next_state, reward, done, info = env.step(action)
        recording_reward.append(reward)

        state = next_state

        if done:
            if frame == (MAX_FRAMES - 1):
                success = 1
                print("Done. Episode {} finished after {} timesteps".format(trial, frame + 1))
            else:
                print('failed in frame:', frame)
                trial = trial - 1
            break
        elif frame == (MAX_FRAMES - 1):
            success = 1
            print("Episode {} finished after {} timesteps".format(trial, frame + 1))
    else:
        env.stats_recorder.save_complete()
        env.stats_recorder.done = True

    if success == 1:
        recording_obs = np.array(recording_obs, dtype=np.float16)
        recording_state = np.array(recording_state, dtype=np.float16)
        recording_action = np.array(recording_action, dtype=np.float16)
        recording_reward = np.array(recording_reward, dtype=np.float16)

        filename = '{}/{}/trail_{}.npz'.format(dataset_name, cartpole_version, trial)
        np.savez_compressed(filename,
                            obs=recording_obs,
                            state=recording_state,
                            action=recording_action,
                            reward=recording_reward)

    trial += 1

env.close()
env.env.close()