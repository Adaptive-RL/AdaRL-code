import gym
import numpy as np

ball_position = []

new_back = {
    'R': np.array([255, 0, 0]),
    'G': np.array([0, 255, 0]),
    'W': np.array([255, 255, 255]),
}


class PONG():
    def __init__(self, params):
        self.env = gym.make("Pong-v0")

        self.back = np.array([144, 72, 17])
        self.ball = np.array([236, 236, 236])
        self.back_type = params['back_type']
        self.scalar = params['scalar']
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.observation_space_shape = (210, 160, 3,)
        self.state = np.zeros(self.observation_space_shape)

    def render(self):
        self.env.render()

    def get_state(self, n_obs):
        target = np.zeros_like(n_obs)

        for i in range(n_obs.shape[0]):
            for j in range(n_obs.shape[1]):
                if (n_obs[i, j] == self.back).all():
                    if self.back_type in new_back:
                        target[i, j] = new_back[self.back_type]
                    else:
                        target[i, j] = n_obs[i, j]

                elif (n_obs[i, j] == self.ball).all():
                    if 190 > i > 33:
                        ball_position.append((i, j))
                    target[i, j] = n_obs[i, j]
                else:
                    target[i, j] = n_obs[i, j]

        if len(ball_position) > 0:

            x = sum([i for i, _ in ball_position]) / len(ball_position) + 0.5
            y = sum([j for _, j in ball_position]) / len(ball_position) + 0.5

            for i in range(max(33, int(x - 2 * self.scalar)),
                           min(190, int(max(x + 2 * self.scalar, 1 + x - 2 * self.scalar)))):
                for j in range(max(0, int(y - self.scalar)), min(160, int(max(y + self.scalar, 1 + y - self.scalar)))):
                    target[int(i), int(j)] = self.ball
        self.state = target

        return self.state

    def reset(self):
        self.state = np.zeros_like((210, 160, 3))

        obs = self.env.reset()
        return self.get_state(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return self.get_state(obs), rew, done, info

    def close(self):
        return

    def display_grid(self):
        return self.state
