import numpy as np
import torch

from .base import DmControlWrapper


# reproduction for PILCO
class DmCartPoleSwingUpEnv(DmControlWrapper):

    def __init__(self,
                 episode_length=25,
                 balance=False,
                 noise_k=0.01,
                 *args,
                 **kwargs):
        if balance:
            name = 'balance'
        else:
            name = 'swingup'
        # noise for [x, theta, xdot, thetadot]
        self.obs_noise = np.array(
            [0.01, 1.0 / 180 * np.pi, 0.1, 10 / 180 * np.pi]) * noise_k
        super().__init__('cartpole',
                         name,
                         episode_length,
                         control_timestep=0.1)

        init_angle = 0.0 if balance else np.pi
        self.init_state = np.array([0.0, init_angle, 0.0, 0.0])

    def _add_obs_noise(self, state):
        return state + self.obs_noise * np.random.randn(*state.shape)

    def _get_observation(self):
        position, angle = self.env.physics.position()
        velocity, angular_velocity = self.env.physics.velocity()
        state = np.array([position, angle, velocity, angular_velocity],
                         dtype=np.float32)
        observation = self._add_obs_noise(state)
        return observation

    def compute_reward(self, obs, act):
        weights = np.array([[0.5, 0.5]])
        target = np.array([0.0, 0.0])
        dist = (((obs[:2] - target) * weights)**2).sum()
        return (1.0 - np.exp(-dist))

    def compute_reward_torch(self, obs, act):
        weights = torch.tensor([[0.5, 0.5]],
                               dtype=torch.float32,
                               device=obs.device)
        target = torch.tensor([[0.0, 0.0]],
                              dtype=torch.float32,
                              device=obs.device)
        dist = (((obs[:, :2] - target) * weights)**2).sum(dim=1)
        return (1.0 - torch.exp(-dist))
