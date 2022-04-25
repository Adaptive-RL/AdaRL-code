import gym
import gym_cartpole_world
import numpy as np


MAX_FRAMES = 4
MAX_TRIALS = 100
env = gym.make('CartPoleWorld-v0')
env.initialize()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

theta_threshold = env.theta_threshold
x_threshold = env.x_threshold


trial = 0
while trial < MAX_TRIALS:
    success = 0

    state = env.reset()

    for frame in range(MAX_FRAMES):
        # add obs
        obs = env.render(mode='rgb_array')

        # add action
        state = np.reshape(state, [1, state_size])
        action = 1 if np.random.rand() > 0.5 else 0

        # add reward
        next_state, reward, done, info = env.step(action)

        state = next_state

        if done:
            if frame == (MAX_FRAMES - 1):
                success = 1
                print("Done. Episode {} finished after {} timesteps".format(trial, frame + 1))
            else:
                trial = trial - 1
            break
        elif frame == (MAX_FRAMES - 1):
            success = 1
            print("Episode {} finished after {} timesteps".format(trial, frame + 1))

    trial += 1

env.close()
