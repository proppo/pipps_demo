import numpy as np
import cv2
import gym
import torch
import copy

from gym import spaces
from dm_control import suite


class DmControlWrapper(gym.Env):

    def __init__(self, domain_name, task_name, episode_length, **kwargs):
        self.env = suite.load(domain_name=domain_name,
                              task_name=task_name,
                              environment_kwargs=kwargs)

        # dm_control specs
        self.observation_spec = self.env.observation_spec()
        self.action_spec = self.env.action_spec()

        # check observation shape
        obs = self.reset()
        observation_size = obs.shape[0]

        # gym spaces
        self.observation_space = spaces.Box(low=-1.0,
                                            high=1.0,
                                            shape=(observation_size, ))
        self.action_space = spaces.Box(low=self.action_spec.minimum,
                                       high=self.action_spec.maximum,
                                       shape=self.action_spec.shape)

        self.episode_length = episode_length
        self.episode_step = 0

    def render(self, mode='human'):
        image = self.env.physics.render(480, 640, camera_id=0)
        # BGR to RGB conversion
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mode == 'rgb_array':
            return image
        else:
            cv2.imshow('render', image)
            cv2.waitKey(10)

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._get_observation()

        # if compute_reward is not defined, use original reward
        rew = self.compute_reward(obs, action)
        if rew is None:
            rew = time_step.reward

        self.episode_step += 1

        ter = self.episode_step == self.episode_length

        # mujoco physics state to reproduce identical states later
        state = self.get_physics_state()

        return obs, rew, ter, {'state': copy.deepcopy(state)}

    def reset(self):
        self.episode_step = 0
        self.env.reset()
        return self._get_observation()

    def compute_reward(self, obs, action):
        return None

    def compute_reward_torch(self, obs):
        raise NotImplementedError

    def _get_observation(self):
        raise NotImplementedError

    def get_physics_state(self):
        return self.env.physics.get_state()

    def set_physics_state(self, state):
        self.env.physics.set_state(state)
        self.env.physics.forward()
