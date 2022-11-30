"""
This code is a modified version of https://github.com/zuoxingdong/DeepPILCO/blob/master/cartpole_swingup.py.
"""

import numpy as np
import math
import gym
import torch
import copy

from gym import spaces
from gym.utils import seeding


class CartPoleSwingUpEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    def __init__(self,
                 episode_length=25,
                 balance=False,
                 noise_k=0.01,
                 tip_cost=False,
                 init_noise_std=0.02):
        self.g = 9.82  # gravity
        self.m_c = 0.5  # cart mass
        self.m_p = 0.5  # pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.l = 0.5  #0.6  # pole's length
        self.m_p_l = (self.m_p * self.l)
        self.force_mag = 10.0
        self.num_int_steps = 10  # number of steps used to integrate up to time dt
        self.dt = 0.1  # seconds between state updates
        self.inner_dt = self.dt / self.num_int_steps
        self.b = 0.1  # friction coefficient

        # noise for [x, theta, xdot, thetadot]
        self.obs_noise = np.array(
            [0.01, 1.0 / 180 * np.pi, 0.1, 10 / 180 * np.pi]) * noise_k

        # Same for all dimensions
        self.init_noise_std = init_noise_std

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max
        ])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1, ))
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.episode_step = 0
        self.episode_length = episode_length

        init_angle = 0.0 if balance else np.pi
        self.init_state = np.array([0.0, init_angle, 0.0, 0.0])

        if tip_cost:
            self.compute_reward = self.tip_cost
            self.compute_reward_torch = self.tip_cost_torch

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def add_obs_noise(self, state):
        return state + self.obs_noise * np.random.randn(*state.shape)

    def step(self, action):
        # Valid action
        action = np.clip(action * self.force_mag, -self.force_mag,
                         self.force_mag)[0]

        state = self.state
        x, theta, x_dot, theta_dot = state
        dt = self.inner_dt

        for _ in range(self.num_int_steps):
            # Simple Euler integration.
            s = math.sin(theta)
            c = math.cos(theta)

            xdot_update = (-2 * self.m_p_l *
                           (theta_dot**2) * s + 3 * self.m_p * self.g * s * c +
                           4 * action - 4 * self.b * x_dot) / (
                               4 * self.total_m - 3 * self.m_p * c**2)
            thetadot_update = (-3 * self.m_p_l * (theta_dot**2) * s * c + 6 * self.total_m * self.g * \
                               s + 6 * (action - self.b * x_dot) * c) / (4 * self.l * self.total_m - 3 * self.m_p_l * c**2)
            x = x + x_dot * dt
            theta = theta + theta_dot * dt
            x_dot = x_dot + xdot_update * dt
            theta_dot = theta_dot + thetadot_update * dt

        self.state = (x, theta, x_dot, theta_dot)

        self.episode_step += 1
        ter = self.episode_step == self.episode_length

        rew = self.compute_reward(np.array(self.state), action)

        observation = self.add_obs_noise(np.array(self.state))
        return observation, rew, ter, {'state': copy.deepcopy(state)}

    def compute_reward(self, obs, act):
        weights = np.array([1.0, 1.0])
        target = np.array([0.0, 0.0])
        dist = (((obs[:2] - target) / weights)**2).sum()
        return (1.0 - np.exp(-0.5 * dist))

    def compute_reward_torch(self, obs, act):
        weights = torch.tensor([[1.0, 1.0]],
                               dtype=torch.float32,
                               device=obs.device)
        target = torch.tensor([[0.0, 0.0]],
                              dtype=torch.float32,
                              device=obs.device)
        dist = (((obs[:, :2] - target) / weights)**2).sum(dim=1)
        return (1.0 - torch.exp(-0.5 * dist))

    def tip_cost(self, obs, act):
        # obs dimensions are cart_pos, angle, cart_vel angle_vel
        x, theta, _, _ = obs
        s = math.sin(theta)
        c = math.cos(theta)
        trigobs = np.hstack([obs, s, c])

        cw = 0.5  # width
        ell = self.l
        q = np.zeros([6, 6])
        q += np.array([[1, 0, 0, 0, ell, 0]]).T @ np.array(
            [[1, 0, 0, 0, ell, 0]])
        q[5, 5] = ell**2  # final dimension with cos
        q = q / (cw**2)

        target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        diff = trigobs - target
        dist = diff.T @ q @ diff

        return (1.0 - np.exp(-0.5 * dist))

    def tip_cost_torch(self, obs, act):
        # obs dimensions are cart_pos, angle, cart_vel angle_vel
        angle_dim = 1
        trigobs = torch.cat([
            obs,
            torch.sin(obs[:, angle_dim:angle_dim + 1]),
            torch.cos(obs[:, angle_dim:angle_dim + 1])
        ],
                            dim=1)

        cw = 0.5  # width
        ell = self.l
        lvec = torch.tensor([[1, 0, 0, 0, ell, 0]], device=obs.device)
        q = torch.zeros(6, 6, device=obs.device)
        q += lvec.T @ lvec
        q[5, 5] = ell**2  # final dimension with cos
        q = q / (cw**2)

        target = torch.zeros(1, 6, device=obs.device)
        target[0, 5] = 1.0
        diff = trigobs - target

        dist = diff @ q
        dist = (dist * diff).sum(dim=1)

        return (1.0 - torch.exp(-0.5 * dist))

    def reset(self, state=None):
        if state is None:
            self.state = self.np_random.normal(self.init_state,
                                               self.init_noise_std)
        else:
            self.state = state
        self.steps_beyond_done = None
        self.episode_step = 0
        observation = self.add_obs_noise(np.array(self.state))
        return observation

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 1200
        screen_height = 800

        world_width = 5  # max visible position of cart
        scale = screen_width / world_width
        carty = 200  # TOP OF CART
        polewidth = 12.0
        polelen = scale * self.l  # 0.5 or self.l
        cartwidth = 80.0
        cartheight = 40.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2

            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(0.7, 0.0, 0.0)
            self.viewer.add_geom(cart)

            l, r, t, b = -polewidth / 2, polewidth / \
                2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0, 0.5, 0.7)
            self.poletrans = rendering.Transform(translation=(0, 0))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.1, 1, 1)
            self.viewer.add_geom(self.axle)

            # Make another circle on the top of the pole
            self.pole_bob = rendering.make_circle(polewidth / 2)
            self.pole_bob_trans = rendering.Transform()
            self.pole_bob.add_attr(self.pole_bob_trans)
            self.pole_bob.add_attr(self.poletrans)
            self.pole_bob.add_attr(self.carttrans)
            self.pole_bob.set_color(0, 0, 0)
            self.viewer.add_geom(self.pole_bob)

            self.wheel_l = rendering.make_circle(cartheight / 4)
            self.wheel_r = rendering.make_circle(cartheight / 4)
            self.wheeltrans_l = rendering.Transform(
                translation=(-cartwidth / 2, -cartheight / 2))
            self.wheeltrans_r = rendering.Transform(translation=(cartwidth / 2,
                                                                 -cartheight /
                                                                 2))
            self.wheel_l.add_attr(self.wheeltrans_l)
            self.wheel_l.add_attr(self.carttrans)
            self.wheel_r.add_attr(self.wheeltrans_r)
            self.wheel_r.add_attr(self.carttrans)
            self.wheel_l.set_color(0, 0, 0)  # Black, (B, G, R)
            self.wheel_r.set_color(0, 0, 0)  # Black, (B, G, R)
            self.viewer.add_geom(self.wheel_l)
            self.viewer.add_geom(self.wheel_r)

            self.track = rendering.Line(
                (0, carty - cartheight / 2 - cartheight / 4),
                (screen_width, carty - cartheight / 2 - cartheight / 4))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[1])
        self.pole_bob_trans.set_translation(-self.l * np.sin(x[1]),
                                            self.l * np.cos(x[1]))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_physics_state(self):
        return self.state

    def set_physics_state(self, state):
        self.state = state
