import random
import numpy as np
from collections import deque

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

import gym_cartpole_world
import gym_pong


# Double DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQNAgent:
    def __init__(self, state_size, action_size, theta_size, is_load=False, load_p=None):
        # if you want to see learning, then change to True
        self.render = False
        self.load_model = is_load
        if self.load_model:
            self.load_path = load_p
        # get size of state and action and also theta
        self.state_size = state_size
        self.action_size = action_size
        self.theta_size = theta_size

        # these is hyper parameters for the Double DQN
        self.discount_factor = 1
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.5
        self.batch_size = 640
        self.train_start = 5000
        self.update_target_steps = 10000
        # create replay memory using deque
        self.memory = deque(maxlen=200000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights(self.load_path)

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    # note theta is also the input!!!
    def build_model(self):
        model = Sequential()
        # note that we also need to input theta
        model.add(Dense(128, input_dim=self.state_size + self.theta_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, clipnorm=1))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state, theta, is_training=True):
        if is_training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state_all = np.concatenate((state, theta), axis=1)
            q_value = self.model.predict(state_all)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, theta, done, score):
        self.memory.append((state, action, reward, next_state, theta, done))
        if score >= 50 and score < 100:
            self.epsilon_min = 0.3
        if score >= 100 and score < 200:
            self.epsilon_min = 0.2
        if score >= 200 and score < 300:
            self.epsilon_min = 0.1
        if score >= 300 & score < 450:
            self.epsilon_min = 0.01
        if score >= 450:
            self.epsilon_min = 0.001
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        update_input = np.zeros((self.batch_size, self.state_size + self.theta_size))
        update_target = np.zeros((self.batch_size, self.state_size + self.theta_size))
        action, reward, done = [], [], []
        mini_batch = random.sample(self.memory, self.batch_size)

        for i in range(self.batch_size):
            update_input_tmp = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target_tmp = mini_batch[i][3]
            theta_tmp = mini_batch[i][4]
            update_input[i] = np.concatenate((update_input_tmp, theta_tmp), axis=1)
            update_target[i] = np.concatenate((update_target_tmp, theta_tmp), axis=1)
            done.append(mini_batch[i][5])

        target = self.model.predict(update_input)  # Q value
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (target_val[i][a])
        estModel = self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)
        train_loss = estModel.history['loss']
        return np.mean(train_loss)
