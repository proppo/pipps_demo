import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import gpytorch

from torch.optim import Adam
from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ZeroMean, MultitaskMean, ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.settings import cholesky_jitter, fast_pred_var


class GPModel(gpytorch.models.ExactGP):

    def __init__(self, train_inputs, train_outputs, likelihood):
        super().__init__(train_inputs, train_outputs, likelihood)
        dim = train_outputs.shape[1]
        inp_dim = train_inputs.shape[1]
        self.mean_module = MultitaskMean(ZeroMean(), num_tasks=dim)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([dim]),
                                       ard_num_dims=inp_dim),
            batch_shape=torch.Size([dim]))

    def forward(self, x):
        mean_x = self.mean_module(x).transpose(1, 0)
        covar_x = self.covar_module(x)
        dist = MultivariateNormal(mean_x, covar_x)

        return MultitaskMultivariateNormal.from_batch_mvn(dist)


def enable_cholesky():
    return gpytorch.settings.fast_computations(covar_root_decomposition=False,
                                               log_prob=False,
                                               solves=False)


class DynamicsModel:

    def __init__(self):
        self.train_x = []
        self.train_y = []
        self.device = 'cpu:0'
        self.model = None

    def to_cuda(self):
        self.device = 'cuda:0'
        self.model.to(self.device)

    def to_cpu(self):
        self.device = 'cpu:0'
        self.model.to(self.device)

    def append(self, train_x, train_y):
        self.train_x += train_x
        self.train_y += train_y

    def train(self, train_x, train_y, num_updates):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def save(self, logger, epoch):
        logger.add_model('model', epoch, self.model)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def eval(self):
        self.model.eval()

    def print_properties(self):
        print('Model property printing not implemented.')


class GPDynamicsModel(DynamicsModel):

    def __init__(
            self,
            train_x,
            train_y,
            lr,
            noise_constraint=GreaterThan(1e-6),  #1e-4
            jitter=1e-6):
        super().__init__()
        self.lr = lr
        self.jitter = jitter

        train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
        train_y_tensor = torch.tensor(train_y, dtype=torch.float32)

        dim = train_y_tensor.shape[1]

        self.likelihood = MultitaskGaussianLikelihood(
            num_tasks=train_y[0].shape[0], noise_constraint=noise_constraint)

        self.model = GPModel(train_x_tensor, train_y_tensor, self.likelihood)

        self.append(train_x, train_y)

    def smart_param_init(self, train_x_tensor, train_y_tensor):
        """Smart initialization for lengthscale, outputscale and
        noise hyperparameters.

        """
        dim = train_y_tensor.shape[1]
        dim_inp = train_x_tensor.shape[1]
        # Lengthscales are initialized at roughly 1/10 of the std of the input data spread.
        self.model.covar_module.base_kernel.raw_lengthscale = \
            torch.nn.Parameter(
                self.model.covar_module.base_kernel.raw_lengthscale_constraint._inv_transform(
                train_x_tensor.std(axis=0)/10).repeat([dim,1]).reshape([dim, 1, dim_inp])
                )
        # Outputscale variance is initialized at output variance of the target data
        self.model.covar_module.base_kernel.raw_outputscale = \
            torch.nn.Parameter(
                self.model.covar_module.raw_outputscale_constraint._inv_transform(
                    train_y_tensor.var(axis=0))
                )
        # The noise has two components, the top one is a shared parameter that
        # is added to all outputs, the bottom one is a specific noise for each
        # separate model. The shared noise is initialized to 1/1000 of the output variance,
        # the separate noises are initialized at the output variance. Ideally,
        # I would not want the shared parameter at all, but it seems to be added in
        # GPyTorch by default.
        self.likelihood.raw_noise = \
            torch.nn.Parameter(
                self.likelihood.raw_noise_constraint._inv_transform(
                    train_y_tensor.var(axis=0).mean()/1000)
                )
        self.likelihood.noise_covar.raw_noise = \
            torch.nn.Parameter(
                self.likelihood.noise_covar.raw_noise_constraint._inv_transform(
                    train_y_tensor.var(axis=0))
                )

    def predict(self, x, t):
        with enable_cholesky(), fast_pred_var(), cholesky_jitter(self.jitter):
            dist = self.model(x)
            pred = self.likelihood(dist)

        # Constrain variances to be positive
        variance = pred.variance
        mean = pred.mean

        # while torch.any(variance < 0):
        #     variance[variance < 0] += self.jitter
        variance[variance < 0] = variance[variance < 0] - \
            variance[variance < 0].detach() + self.jitter

        return mean, variance

    def train(self, train_x, train_y, num_opt_steps):
        self.append(train_x, train_y)

        train_x_tensor = torch.tensor(self.train_x,
                                      dtype=torch.float32,
                                      device=self.device)
        train_y_tensor = torch.tensor(self.train_y,
                                      dtype=torch.float32,
                                      device=self.device)

        # Smart initialization of lengthscale, outputscale
        # and noise hyperparameters. This step is crucial.
        self.smart_param_init(train_x_tensor, train_y_tensor)

        self.model.train()
        self.likelihood.train()

        loss_history = []

        # build optimizer every time
        optimizer = Adam(self.model.parameters(), lr=self.lr)

        with enable_cholesky(), cholesky_jitter(self.jitter):
            # loss for GP
            mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

            # update data of GP
            self.model.set_train_data(train_x_tensor,
                                      train_y_tensor,
                                      strict=False)

            for _ in range(num_opt_steps):
                pred = self.model(train_x_tensor)
                loss = -mll(pred, train_y_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_history.append(loss.detach().cpu().numpy())

        self.model.eval()
        self.likelihood.eval()

        self.print_properties()

        return loss_history

    def print_properties(self):
        #print('Noise hyperparameters:', self.)
        print('Lengthscale hyperparameters:',
              self.model.covar_module.base_kernel.lengthscale)
        print('Noise hyperparameters:', self.model.likelihood.noise,
              self.model.likelihood.noise_covar.noise)
        print('Noise level stds are:',
              (self.model.likelihood.noise +
               self.model.likelihood.noise_covar.noise).sqrt())
        print('Output scale parameter stds are:',
              self.model.covar_module.outputscale.sqrt())
        #print('All params are:', [param for param in self.model.named_parameters()])
