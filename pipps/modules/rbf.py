import numpy as np
import torch.nn as nn
import torch

from proppo.modules import BatchedLinear
from proppo.utils import expand


class RBF(nn.Module):
    """ Radial basis function neural network with batched parameters.

    Attributes:
        batch_size (int): batch size for parameters.
        input_size (int): input size of this function.
        centers (torch.nn.Parameter): center matrix
        lam (torch.nn.Parameter): weight parameters.
        fc (proppo.modules.BatchedLinear): projection layer to output.

    """

    def __init__(self,
                 input_size,
                 output_size,
                 batch_size,
                 num_centers,
                 center_mu=0.0,
                 center_std=1.0):
        """ __init__ method.

        Arguments:
            input_size (int): input size.
            output_size (int): output size.
            batch_size (int): batch size.
            num_centers (int): number of center units.
            center_mu (float or torch.Tensor):
                mean of center units. Mean values of each feature can be
                designated with torch.Tensor with shape of (input_size,)
            center_std (float or torch.Tensor):
                standard deviation of center units. Standard deviation values
                of each feature can be designated with torch.Tensor with shape
                of (input_size,)

        """
        super().__init__()
        self.batch_size = batch_size
        self.input_size = input_size

        if isinstance(center_mu, (np.ndarray, list)):
            center_mu = torch.tensor(center_mu, dtype=torch.float32)

        if isinstance(center_mu, torch.Tensor) and center_mu.ndim == 1:
            center_mu = center_mu.view(1, -1)

        if isinstance(center_std, torch.Tensor) and center_std.ndim == 1:
            center_std = center_std.view(1, -1)

        norm_noise = torch.randn(num_centers, input_size)
        self.centers = nn.Parameter(center_mu + norm_noise * center_std)
        self.lam = nn.Parameter(torch.log(torch.ones(1, input_size)))
        self.fc = BatchedLinear(num_centers,
                                output_size,
                                batch_size,
                                bias=False)

        self.rebuild_batched_parameters()

    def forward(self, x, use_batched_parameters=True):
        """ forward method.

        Arguments:
            x (torch.Tensor): input tensor.
            use_batched_parameters (bool): flag to use batched parameters.

        Returns:
            torch.Tensor: output tensor.

        """
        if use_batched_parameters:
            distance = x.view(self.batch_size, 1, -1) - self.batched_centers
            distance = distance * torch.exp(self.batched_lam)
            norm = torch.norm(distance, dim=2)**2
            h = torch.exp(-norm)
            return self.fc(h)
        else:
            distance = x.view(-1, 1, self.input_size) - self.centers
            distance = distance * torch.exp(self.lam)
            norm = torch.norm(distance, dim=2)**2
            h = torch.exp(-norm)
            return self.fc(h, use_batched_parameters=False)

    def get_batched_parameters(self):
        """ Returns batched parameters.

        Returns:
            list: list of batched weights and parameters.

        """
        parameters = self.fc.get_batched_parameters()
        parameters += [self.batched_centers, self.batched_lam]
        return parameters

    def rebuild_batched_parameters(self):
        """ Rebuild batched parameters from the latest non-batched parameters.

        This method should be called after the non-batched parameters are
        updated.

        """
        self.fc.rebuild_batched_parameters()
        self.batched_centers = expand(self.centers, self.batch_size)
        self.batched_lam = expand(self.lam, self.batch_size)
