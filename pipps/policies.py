import numpy as np
import torch.nn as nn
import torch

from torch.optim import Adam
from torch.distributions import Normal
from pipps.modules.rbf import RBF


def sins_activation(h):
    return (9 * torch.sin(h) + torch.sin(3 * h)) / 8


class RBFPolicyModel(nn.Module):

    def __init__(self,
                 obs_size,
                 act_size,
                 batch_size,
                 num_centers,
                 init_pos,
                 activation='tanh'):
        super().__init__()
        self.rbf = RBF(obs_size, act_size, batch_size, num_centers, [init_pos])
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sins':
            self.activation = sins_activation
        else:
            raise NotImplementedError(activation +
                                      ' activation is not implemented.')

    def forward(self, x, use_batched_parameters=True):
        h = self.rbf(x, use_batched_parameters)
        return self.activation(h)

    def get_batched_parameters(self):
        return self.rbf.get_batched_parameters()

    def rebuild_batched_parameters(self):
        self.rbf.rebuild_batched_parameters()


class Policy:

    def __init__(self, obs_size, act_size, batch_size, lr):
        self.device = 'cpu:0'
        self.model = None
        self.obs_size = obs_size
        self.act_size = act_size
        self.batch_size = batch_size
        self.lr = lr

    def to_cuda(self):
        self.device = 'cuda:0'
        self.model.to(self.device)

    def to_cpu(self):
        self.device = 'cpu:0'
        self.model.to(self.device)

    @property
    def batched_parameters(self):
        return self.model.get_batched_parameters()

    @property
    def parameters(self):
        return self.model.parameters

    def rebuild_batched_parameters(self):
        self.model.rebuild_batched_parameters()

    def compute_action(self, x):
        raise NotImplementedError

    def save(self, logger, epoch):
        logger.add_model('policy', epoch, self.model)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def eval(self):
        self.model.eval()


class RBFPolicy(Policy):

    def __init__(self,
                 obs_size,
                 act_size,
                 batch_size,
                 lr,
                 num_centers,
                 init_pos=[0, np.pi, 0, 0],
                 activation='tanh'):
        super().__init__(obs_size, act_size, batch_size, lr)
        self.model = RBFPolicyModel(obs_size, act_size, batch_size,
                                    num_centers, init_pos, activation)

    def compute_action(self, x, use_batched_parameters=True):
        return self.model(x, use_batched_parameters=use_batched_parameters)
