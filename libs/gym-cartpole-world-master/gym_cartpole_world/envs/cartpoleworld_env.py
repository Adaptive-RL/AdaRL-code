# -*- coding: utf-8 -*-
"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
Modified by Aaditya Ravindran to include friction and random sensor & actuator noise
"""
import cv2
import logging
import math
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)


class CartPoleWorldEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, case=1, gravity=9.8, masscart=1.0, noises=0, polelen=0.5):
        self.__version__ = "0.1.0"
        self.theta_threshold = None
        self.x_threshold = None
        self.theta_threshold_radians = None
        self.init_indicator = False

        self.theta_threshold_max = 45
        self.x_threshold_max = 5

        print("CartPoleWorldEnv - Version {}".format(self.__version__))

        self.gravity = gravity
        self.masscart = masscart
        self.noises = noises
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = polelen  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self._seed()
        if case < 4:
            self.force_mag = 10.0 * (1 + self.addnoise(case))
            self.case = 1
        else:
            self.force_mag = 10.0
            self.case = case

        self.tau = 0.02  # seconds between state updates
        self.frictioncart = 5e-4  # AA Added cart friction
        self.frictionpole = 2e-6  # AA Added cart friction

        self.action_space = spaces.Discrete(2)  # AA Set discrete states back to 2
        self.observation_space = None

        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def initialize(self, theta_threshold=45, x_threshold=5):
        self.init_indicator = True

        if theta_threshold > self.theta_threshold_max:
            logger.error("Error: theta_threshold should not exceed {}".format(self.theta_threshold_max))
            return
        else:
            self.theta_threshold = theta_threshold

        if x_threshold > self.x_threshold_max:
            logger.error("Error: x_threshold should not exceed {}".format(self.x_threshold_max))
            return
        else:
            self.x_threshold = x_threshold

        print("CartPoleWorldEnv - theta_threshold: {}, x_threshold: {}".format(self.theta_threshold, self.x_threshold))

        # Angle at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # self.x_threshold = 2.4

        self.theta_threshold_radians = self.theta_threshold * 2 * math.pi / 360
        # self.x_threshold = 20

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])
        self.observation_space = spaces.Box(-high, high)

    def addnoise(self, x):
        return {
            1: 0,
            2: self.np_random.uniform(low=-0.05, high=0.05, size=(1,)),  # 5% actuator noise
            3: self.np_random.uniform(low=-0.10, high=0.10, size=(1,)),  # 10% actuator noise
            4: self.np_random.uniform(low=-0.05, high=0.05, size=(1,)),  # 5% sensor noise
            5: self.np_random.uniform(low=-0.10, high=0.10, size=(1,)),  # 10% sensor noise
            6: self.np_random.normal(loc=0, scale=np.sqrt(0.10), size=(1,)),  # 0.1 var sensor noise
            7: self.np_random.normal(loc=0, scale=np.sqrt(0.20), size=(1,)),  # 0.2 var sensor noise
        }.get(x, 1)

    def _seed(self, seed=None):  # Set appropriate seed value
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        # AA Added cart friction
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta - self.frictioncart * np.sign(
            x_dot)) / self.total_mass
        # AA Added pole friction
        thetaacc = (self.gravity * sintheta - costheta * temp - self.frictionpole * theta_dot / self.polemass_length
                    ) / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        noise = self.addnoise(self.case)
        x = (x + self.tau * x_dot)
        x_dot = (x_dot + self.tau * xacc)
        theta = (theta + self.tau * theta_dot) * (1 + noise)
        theta_dot = (theta_dot + self.tau * thetaacc)
        self.state = (x, x_dot, theta, theta_dot)
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning(
                    "You are calling 'step()' even though this environment has already returned done = True. "
                    "You should always call 'reset()' once you receive 'done = True' -- "
                    "any further steps are undefined behavior.")

            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):

        if not self.init_indicator:
            logger.error("You should always call 'initialize()' before calling 'reset()'.")

        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 400
        screen_height = 400

        carty = 100  # TOP OF CART
        polewidth = 10.0
        # polelen = scale * 1.0
        polelen = 125 * 2 * self.length
        cartwidth = 50.0
        cartheight = 30.0

        # world_width = self.x_threshold * 2
        world_width = self.x_threshold_max * 2
        theta_threshold_radians_max = self.theta_threshold_max * 2 * math.pi / 360

        real_screen_width = screen_width - \
                            np.maximum(cartwidth / 2,
                                       (polelen - polewidth / 2) * np.sin(theta_threshold_radians_max)) * 2
        scale = real_screen_width / world_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])
        render_f = self.viewer.render(return_rgb_array=mode == 'rgb_array') # value range [0, 255]
        if self.noises != 0:
            gaussian = np.random.normal(0, self.noises, (screen_height, screen_width))
            gaussian = gaussian[:, :, None] * np.ones(3, dtype=int)[None, None, :]
            render_f = render_f + gaussian
            render_f = np.clip(render_f, 0, 255)  # we might get out of bounds due to noise
            render_f = np.uint8(render_f)
        return render_f


