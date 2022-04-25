import cv2
import numpy as np
import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

import gym
import gym_pong

MAX_FRAMES = 1000
MAX_TRIALS = 10000

# 15 is the length of bar
L = 15 / 2

parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str, required=True)
parser.add_argument('-v', type=str, required=True)
parser.add_argument('-rewardm', type=str, default='linear', help='linear or non-linear')
parser.add_argument('-k1', type=float, default=0.1, help='linear')
parser.add_argument('-k2', type=float, default=2.0, help='non-linear')
args = parser.parse_args()

pong_name = args.name
pong_version = args.v
env_name = pong_name + '-' + pong_version
env = gym.make(env_name)

reward_reg = args.rewardm
alpha_r = args.k1
alpha_r = args.k2
if 'm' not in env_name:
    dataset_name = './dataset/Pong/' + 'R_len{}_episode{}_pong'.format(
        MAX_FRAMES, MAX_TRIALS)
    assert reward_reg in ['linear', 'non-linear']
    if reward_reg == 'linear':
        pong_version = 'v40'
        assert args.k1 in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.5, 0.9]
    else:
        assert args.k2 in [2.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 4.0, 1.0]
        pong_version = 'v41'

else:
    if pong_version[1] == '0':
        dataset_name = './dataset/Pong/' + 'S_len{}_episode{}_pong'.format(
            MAX_FRAMES, MAX_TRIALS)
    elif pong_version[1] == '1':
        dataset_name = './dataset/Pong/' + 'O_len{}_episode{}_pong'.format(
            MAX_FRAMES, MAX_TRIALS)
    elif pong_version[1] == '2':
        dataset_name = './dataset/Pong/' + 'N_len{}_episode{}_pong'.format(
            MAX_FRAMES, MAX_TRIALS)
    elif pong_version[1] == '3':
        dataset_name = './dataset/Pong/' + 'C_len{}_episode{}_pong'.format(
            MAX_FRAMES, MAX_TRIALS)

if not os.path.exists(dataset_name):
    os.makedirs(dataset_name)

if not os.path.exists(os.path.join(dataset_name, pong_version)):
    os.makedirs(os.path.join(dataset_name, pong_version))

for i_episode in range(MAX_TRIALS):
    observation = env.reset()
    obs_iter = []
    action_iter = []
    reward_iter = []
    for t in range(MAX_FRAMES):
        env.render()
        obs = env.render(mode='rgb_array')
        # obs read rgb
        reward_new = 0
        if 'm' not in env_name:
            pos_player = []
            pos_ball = []
            epsilon = 1
            for i in range(obs.shape[0]):
                if i > 33 and i < 194:
                    for j in range(obs.shape[1]):
                        if abs(sum(obs[i][j] - np.array([92, 186, 92]))) == 0:
                            pos_player.append((i, j))
                        if abs(sum(obs[i][j] - np.array([92, 186, 92]))) != 0 and \
                            abs(sum(obs[i][j] - np.array([213, 130, 74]))) != 0 and \
                                abs(sum(obs[i][j] - np.array([144, 72, 17]))) != 0 :
                            pos_ball.append((i, j))
            if len(pos_ball) > 0 and len(pos_player) > 0:
                player_x = np.array([pos[0] for pos in pos_player])
                player_y = np.array([pos[1] for pos in pos_player])
                ball_x = np.array([pos[0] for pos in pos_ball])
                ball_y = np.array([pos[1] for pos in pos_ball])
                ball_x_max = max(ball_x)
                ball_x_min = min(ball_x)
                ball_y_max = max(ball_y)
                player_x_min = min(player_x)
                player_x_max = max(player_x)
                player_y_min = min(player_y)
                touch_d = 0.5 * (max(ball_y) - min(ball_y) + max(player_y) - min(player_y))
                if ball_x_max >= player_x_min and ball_x_max <= (player_x_max + ball_x_max - ball_x_min):
                    if (np.mean(player_y) - np.mean(ball_y)) > 0 and (np.mean(player_y) - np.mean(ball_y) - epsilon) <= touch_d:
                        d = abs(np.mean(player_x) - np.mean(ball_x))
                        if reward_reg == 'non-linear':
                            reward_new = alpha_r * L / (d + 3 * L)
                        elif reward_reg == 'linear':
                            reward_new = alpha_r * d / L
                        else:
                            print('wrong reward func.')
                else:
                    reward_new = 0
        if pong_version[:2] != 'v3':
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (128, 128), interpolation=cv2.INTER_CUBIC)
        action = env.action_space.sample()
        obs_iter.append(obs)
        action_iter.append(action)
        observation, reward, done, info = env.step(action)
        if 'm' not in env_name:
            reward_iter.append(reward_new)
        else:
            reward_iter.append(reward)
        if done:
            if t == 999:
                print("Episode finished after {} timesteps".format(t+1))
            else:
                i_episode = i_episode - 1
    obs_iter = np.array(obs_iter, dtype=np.float16)
    action_iter = np.array(action_iter, dtype=np.float16)
    reward_iter = np.array(reward_iter, dtype=np.float16)

    filename = '{}/{}/trail_{}.npz'.format(dataset_name, pong_version, i_episode)
    np.savez_compressed(filename,
                        obs=obs_iter,
                        action=action_iter,
                        reward=reward_iter)

env.close()