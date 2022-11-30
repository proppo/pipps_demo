import torch

from torch.optim import Adam
from torch.distributions import Normal

from tqdm import trange


class MBAgent:

    def __init__(self,
                 policy,
                 dynamics_model,
                 init_state,
                 reward_func=None,
                 init_noise_std=0.01):
        self.policy = policy
        self.dynamics_model = dynamics_model
        self.init_state = init_state
        self.reward_func = reward_func
        self.init_noise_std = init_noise_std
        self.device = self.policy.device
        self.init_policy_optimizer()

    def init_policy_optimizer(self, adam_eps=1e-4):
        self.optimizer = Adam(self.policy.parameters(),
                              lr=self.policy.lr,
                              eps=adam_eps)

    def compute_action(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32, device=self.policy.device)
            action = self.policy.compute_action(x, False)
            return action.cpu().detach().numpy()

    def simulate(self, rollout_horizon, manager=None):
        # prepare initial observation
        non_batched_obs = torch.tensor(self.init_state,
                                       dtype=torch.float32,
                                       device=self.device)
        noise = torch.randn(self.policy.batch_size,
                            self.policy.obs_size,
                            device=self.device)
        obs = non_batched_obs + self.init_noise_std * noise

        obs_history = [obs]

        total_cost = 0
        # action
        act = self.policy.compute_action(obs)

        for t in range(rollout_horizon):
            # predict next state distribution
            dynamics_input = torch.cat([obs, act], dim=1)
            mean, variance = self.dynamics_model.predict(dynamics_input, t)

            # sampling
            if manager:
                obs = manager.forward(
                    obs + mean,
                    dist_class=Normal,
                    dist_params={
                        'loc': torch.zeros_like(obs),
                        'scale': variance.sqrt()
                    },
                    ivw_target=self.policy.batched_parameters)
            else:
                dist = Normal(obs + mean, variance.sqrt())
                obs = dist.rsample()

            obs_history.append(obs)

            act = self.policy.compute_action(obs)

            # compute costs
            if self.reward_func:
                if manager:
                    cost = manager.append_loss({
                        'obs': obs,
                        'act': act
                    },
                                               lossfunc=self.reward_func)
                else:
                    cost = self.reward_func(obs, act)
                total_cost += cost.view(-1, 1)

        return obs_history, total_cost

    def train_policy(self,
                     rollout_horizon,
                     num_opt_steps,
                     manager=None,
                     callback=None,
                     reinit_optimizer=False):
        assert self.reward_func is not None

        if reinit_optimizer:
            self.init_policy_optimizer()

        total_cost_history = []
        for step in trange(num_opt_steps, smoothing=0.1):
            _, total_cost = self.simulate(rollout_horizon, manager=manager)

            self.optimizer.zero_grad()

            # perform backward
            if manager:
                # compute local returns subtracted by leave-one-out baselines
                manager.backward()
            else:
                total_cost.mean().backward()

            self.optimizer.step()

            # rebuild batched parameters from the latest parameters
            self.policy.rebuild_batched_parameters()

            mean_total_cost = total_cost.mean().cpu().detach().numpy()
            total_cost_history.append(mean_total_cost)

            if callback:
                callback(step, mean_total_cost)

        return total_cost_history

    def train_dynamics_model(self, train_x, train_y, num_updates):
        return self.dynamics_model.train(train_x, train_y, num_updates)

    def to_cuda(self):
        self.policy.to_cuda()
        self.policy.rebuild_batched_parameters()
        self.dynamics_model.to_cuda()
        self.device = 'cuda:0'

    def to_cpu(self):
        self.policy.to_cpu()
        self.policy.rebuild_batched_parameters()
        self.dynamics_model.to_cpu()
        self.device = 'cpu:0'

    def save(self, logger, epoch):
        self.policy.save(logger, epoch)
        self.dynamics_model.save(logger, epoch)
        logger.add_data('dynamics_data', epoch, (self.train_x, self.train_y))

    def load_policy(self, path):
        self.policy.load(path)

    def load_dynamics_model(self, path):
        self.dynamics_model.load(path)

    @property
    def train_x(self):
        return self.dynamics_model.train_x

    @property
    def train_y(self):
        return self.dynamics_model.train_y
