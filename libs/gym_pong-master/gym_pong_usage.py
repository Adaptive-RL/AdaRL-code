import gym
import gym_pong
import cv2
import gym_cartpole_world
import numpy as np
import os

MAX_FRAMES = 1000
MAX_TRIALS = 200

env = gym.make('Pongm-v00')

for i_episode in range(MAX_TRIALS):
    observation = env.reset()
    obs_iter = []
    action_iter = []
    reward_iter = []
    for t in range(MAX_FRAMES):
        env.render()
        obs = env.render(mode='rgb_array')
        action = env.action_space.sample()
        obs_iter.append(obs)
        observation, reward, done, info = env.step(action)
        if done:
            if t == 999:
                print("Episode finished after {} timesteps".format(t+1))
            else:
                t = t - 1
            break

env.close()

