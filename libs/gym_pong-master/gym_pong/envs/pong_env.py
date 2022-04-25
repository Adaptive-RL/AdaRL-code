import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import logging
import math
import random
import numpy as np
import cv2

logger = logging.getLogger(__name__)


try:
    import atari_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you can install Atari dependencies by running 'pip install gym[atari]'.)".format(e))

def to_ram(ale):
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size),dtype=np.uint8)
    ale.getRAM(ram)
    return ram

class AtariEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game='pong', obs_type='image', frameskip=(2, 5), repeat_action_probability=0.,
                 size=1, orientation=0, color='default', noise = 0):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        utils.EzPickle.__init__(self, game, obs_type)
        assert obs_type in ('ram', 'image')

        self.game_path = atari_py.get_game_path(game)
        if not os.path.exists(self.game_path):
            raise IOError('You asked for game %s but path %s does not exist'%(game, self.game_path))
        self._obs_type = obs_type
        self.frameskip = frameskip
        self.size = size
        self.orientation = orientation
        self.color = color
        self.noise = noise
        self.ale = atari_py.ALEInterface()
        self.viewer = None

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        assert isinstance(repeat_action_probability, (float, int)), "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        self.ale.setFloat('repeat_action_probability'.encode('utf-8'), repeat_action_probability)

        self.seed()

        self._action_set = self.ale.getMinimalActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))

        (screen_width,screen_height) = self.ale.getScreenDims()
        if self._obs_type == 'ram':
            self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(128,))
        elif self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b'random_seed', seed2)
        self.ale.loadROM(self.game_path)
        return [seed1, seed2]

    def step(self, a):
        reward = 0.0
        action = self._action_set[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        for _ in range(num_steps):
            reward += self.ale.act(action)
        ob = self._get_obs()

        return ob, reward, self.ale.game_over(), {"ale.lives": self.ale.lives()}

    def _get_image(self):
        return self.ale.getScreenRGB2()

    def _get_ram(self):
        return to_ram(self.ale)

    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        if self._obs_type == 'ram':
            return self._get_ram()
        elif self._obs_type == 'image':
            img = self._get_image()
            (h, w, c) = img.shape
            reduction = math.sqrt(self.size)
            h_s = int(h // reduction)
            w_s = int(w // reduction)
            img = cv2.resize(img, (w_s, h_s))

            if self.orientation != 0:
                (cX, cY) = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D((cX, cY), -self.orientation, 1.0)
                cos = np.abs(M[0, 0])
                sin = np.abs(M[0, 1])

                # compute the new bounding dimensions of the image
                nW = int((h * sin) + (w * cos))
                nH = int((h * cos) + (w * sin))

                # adjust the rotation matrix to take into account translation
                M[0, 2] += (nW / 2) - cX
                M[1, 2] += (nH / 2) - cY
                img = cv2.warpAffine(img, M, (nW, nH), borderValue=(255,255,255))

            if self.color == 'green':
                ratio = [0.51, 2.59, 4.80]
                for i in range(3):
                    img[:,:,i] *= int(ratio[i])
            if self.color == 'red':
                ratio = [1.60, 0.65, 2.40]
                for i in range(3):
                    img[:, :, i] *= int(ratio[i])
            if self.color == 'white':
                ratio = [1.77, 3.59, 17.00]
                for i in range(3):
                    img[:,:,i] *= int(ratio[i])
            if self.color == 'yellow':
                ratio = [1.74, 3.23, 0.53]
                for i in range(3):
                    img[:,:,i] *= int(ratio[i])

            if self.noise != 0:
                gaussian = np.random.normal(0, self.noise, (h, w))
                gaussian = gaussian[:, :, None] * np.ones(3, dtype=int)[None, None, :]
                img = img + gaussian
        img.astype('uint8')
        return img

    # return: (states, observations)
    def reset(self):
        self.ale.reset_game()
        return self._get_obs()

    def render(self, mode='human'):
        img = self._get_image()
        (h, w, c) = img.shape
        reduction = math.sqrt(self.size)
        h_s = int(h // reduction)
        w_s = int(w // reduction)
        #img = img.reshape(h_s, self.size, w_s, self.size, 3).max(3).max(1)  # downsampling
        img = cv2.resize(img, (w_s, h_s))
        #if self.size != 1.:
        #    diff_h = h - h_s
        #    diff_w = w - w_s
        #    color_b = 0
        #    col = np.ones((diff_w, img.shape[0], 3))
        #    img = np.insert(img, img.shape[1], col * color_b, axis=1)  # add cols in right
        #    row = np.ones((diff_h, img.shape[1], 3))
        #    img = np.insert(img, img.shape[0], row * color_b, axis=0)  # add rows in bottom

        if self.orientation != 0:
            (cX, cY) = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D((cX, cY), -self.orientation, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY
            img = cv2.warpAffine(img, M, (nW, nH), borderValue=(255, 255, 255))

        if self.noise != 0:
            gaussian = np.random.normal(0, self.noise, (h, w))
            gaussian = gaussian[:, :, None] * np.ones(3, dtype=int)[None, None, :]
            img = img + gaussian
        if self.color != 'default':
            if self.color == 'green':
                ratio = [0.51, 2.59, 4.80]
                for i in range(3):
                    img[:, :, i] *= int(ratio[i])
            if self.color == 'red':
                ratio = [1.60, 0.65, 2.40]
                for i in range(3):
                    img[:, :, i] *= int(ratio[i])
            if self.color == 'white':
                ratio = [1.77, 3.59, 17.00]
                for i in range(3):
                    img[:, :, i] *= int(ratio[i])
            if self.color == 'yellow':
                ratio = [1.74, 3.23, 0.53]
                for i in range(3):
                    img[:, :, i] *= int(ratio[i])

        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
            'FIRE':    ord(' '),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action

    def clone_state(self):
        """Clone emulator state w/o system state. Restoring this state will
        *not* give an identical environment. For complete cloning and restoring
        of the full state, see `{clone,restore}_full_state()`."""
        state_ref = self.ale.cloneState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_state(self, state):
        """Restore emulator state w/o system state."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreState(state_ref)
        self.ale.deleteState(state_ref)

    def clone_full_state(self):
        """Clone emulator state w/ system state including pseudorandomness.
        Restoring this state will give an identical environment."""
        state_ref = self.ale.cloneSystemState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_full_state(self, state):
        """Restore emulator state w/ system state including pseudorandomness."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreSystemState(state_ref)
        self.ale.deleteState(state_ref)

ACTION_MEANING = {
    0 : "NOOP",
    1 : "FIRE",
    2 : "UP",
    3 : "RIGHT",
    4 : "LEFT",
    5 : "DOWN",
    6 : "UPRIGHT",
    7 : "UPLEFT",
    8 : "DOWNRIGHT",
    9 : "DOWNLEFT",
    10 : "UPFIRE",
    11 : "RIGHTFIRE",
    12 : "LEFTFIRE",
    13 : "DOWNFIRE",
    14 : "UPRIGHTFIRE",
    15 : "UPLEFTFIRE",
    16 : "DOWNRIGHTFIRE",
    17 : "DOWNLEFTFIRE",
}