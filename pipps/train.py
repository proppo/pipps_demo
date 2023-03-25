import numpy as np
import argparse
import pickle
import os
import gym
import pipps.envs
import time

from proppo.modules import get_batched_parameters
from proppo.propagation_manager import PropagationManager
from proppo.propagators import TotalProp, RPProp, LRProp, LossProp, \
    ResampleProp, GRProp, GSLoss
from proppo.baseline_funcs import mean_baseline, no_baseline
from pipps.policies import RBFPolicy
from pipps.dynamics_models import GPDynamicsModel
from pipps.agents import MBAgent
from pipps.logger import prepare_logger
from pipps.utils import resume
from pipps.envs.dm_control.base import DmControlWrapper  # TODO: remove import by default
from pipps.plotting import plot_trajectory_comparison

import torch


def evaluate(env, agent, num_evaluates, render=False):
    episode_rewards = []
    for i in range(num_evaluates):
        obs = env.reset()
        episode_reward = 0.0
        while True:
            act = agent.compute_action([obs])[0]
            obs, rew, ter, _ = env.step(act)
            episode_reward += rew

            if render:
                env.render()

            if ter:
                break

        episode_rewards.append(episode_reward)
    return episode_rewards


def collect_with_random_policy(env, act_size, record=True):
    inputs = []
    outputs = []
    obs = env.reset()
    ter = False
    images = []
    states = []
    while not ter:
        act = np.random.uniform(-1.0, 1.0, size=act_size)
        obs_next, _, ter, info = env.step(act)

        inputs.append(np.hstack([obs, act]))
        outputs.append(obs_next - obs)
        states.append(info['state'])

        if record:
            env.render()
            images.append(env.render('rgb_array'))
            if isinstance(env, DmControlWrapper):
                time.sleep(env.env.control_timestep())
            else:
                time.sleep(env.dt)

        obs = obs_next
    return inputs, outputs, images, states


def collect_with_policy(env, agent, record=True):
    inputs = []
    outputs = []
    obs = env.reset()
    ter = False
    images = []
    states = []
    while not ter:
        act = agent.compute_action([obs])[0]
        obs_next, _, ter, info = env.step(act)

        inputs.append(np.hstack([obs, act]))
        outputs.append(obs_next - obs)
        states.append(info['state'])

        if record:
            env.render()
            images.append(env.render('rgb_array'))
            if isinstance(env, DmControlWrapper):
                time.sleep(env.env.control_timestep())
            else:
                time.sleep(env.dt)

        obs = obs_next
    return inputs, outputs, images, states


def main(args):
    if args.resume:
        start_trial = resume(args.resume, args) + 1
    else:
        start_trial = 0

    if args.num_threads:
        torch.set_num_threads(args.num_threads)

    env_params = {
        'episode_length': args.rollout_horizon,
        'init_noise_std': args.init_noise_std
    }
    if args.noise_k is not None:
        env_params['noise_k'] = args.noise_k

    env = gym.make(args.env, **env_params)

    # observation and action size
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    # data logger
    log_folder_name = args.method
    if args.loss == 'GS':
        log_folder_name = log_folder_name + '_' + 'GS'
    log_folder_name = log_folder_name + '_' + args.id
    dir_path = 'logs/' + args.env
    if args.noise_k is not None:
        dir_path = dir_path + '_k' + str(args.noise_k)
    logger = prepare_logger(dir_path=dir_path, name=log_folder_name)

    # policy function
    policy = RBFPolicy(obs_size,
                       act_size,
                       args.batch_size,
                       args.lr,
                       args.rbf_centers,
                       env.init_state,
                       activation=args.activation)
    if args.load_policy:
        policy.load(args.load_policy)

    # Choose propagation method
    if args.method == 'TP':
        propagator = TotalProp(backprop=False)
    elif args.method == 'LR':
        # By setting backprop=False, it may run into a memory issue,
        # but will usually be computationally a bit faster.
        propagator = LRProp(baseline_func=mean_baseline, backprop=True)
    elif args.method == 'RP':
        propagator = RPProp(backprop=False)
    elif args.method == 'RP_no_detach':
        propagator = RPProp(detach=False)
    elif args.method == 'GR':
        propagator = GRProp()
    elif args.method == 'GR_no_detach':
        propagator = GRProp(detach=False)
    elif args.method == 'No_proppo':
        propagator = None

    if args.loss == 'Regular':
        loss_propagator = None
    elif args.loss == 'GS':
        if args.method == 'LR':
            # TODO: this could be cleaned up
            loss_propagator = GSLoss(gs_prop={
                'shaped_grad': False,
                'backprop': False
            },
                                     gs_loss={
                                         'sumloss': False,
                                         'loss_name': 'shaped_loss'
                                     },
                                     skip={'skip': 2})
        else:
            loss_propagator = GSLoss()

    # PropagationManager with different propagators.
    if args.method == 'RP_no_detach' or args.method == 'GR_no_detach':
        manager = PropagationManager(default_propagator=propagator,
                                     loss_propagator=LossProp(backprop=False))
    elif propagator:
        manager = PropagationManager(default_propagator=propagator,
                                     loss_propagator=loss_propagator)
    else:
        manager = None

    if args.load_dynamics_data:
        # load from snapshot
        with open(args.load_dynamics_data, 'rb') as f:
            train_x, train_y = pickle.load(f)
    else:
        # random trial
        train_x = []
        train_y = []
        for n_trial in range(args.num_random_trials):
            x, y, images, states = collect_with_random_policy(
                env, act_size, args.record)
            train_x += x
            train_y += y
            if args.record:
                logger.add_video('random_episode_video', 0, n_trial, images)
            logger.add_data('physics_states', 0, states)

    # dynamics model
    if args.dynamics_model == 'gp':  # Gaussian Process Model
        dynamics_model = GPDynamicsModel(train_x,
                                         train_y,
                                         args.dynamics_lr,
                                         jitter=args.jitter)
    if args.load_dynamics:
        dynamics_model.load(args.load_dynamics)

    agent = MBAgent(policy=policy,
                    dynamics_model=dynamics_model,
                    init_state=env.init_state,
                    reward_func=env.compute_reward_torch,
                    init_noise_std=args.init_noise_std)

    # move to GPU memory
    if args.gpu:
        agent.to_cuda()

    # save hyperparameters
    logger.add_params(vars(args))

    # save githash
    if args.save_git:
        logger.save_githash()

    train_x = []
    train_y = []
    # learned trials
    for n_trial in range(start_trial, args.num_learned_trials):
        if n_trial > 0 or args.load_policy:
            # collect new trajectories
            train_x, train_y, images, states = collect_with_policy(
                env, agent, args.record)
            if args.record:
                logger.add_video('policy_episode_video', n_trial, 1, images)
            logger.add_data('physics_states', n_trial, states)

        # plot trajectory comparison with model predictions
        if args.plot_traj and n_trial > 0:
            with torch.no_grad():
                obs, _ = agent.simulate(args.rollout_horizon)
            obs_mean = [
                torch.mean(ob, dim=0).detach().cpu().numpy() for ob in obs
            ]
            obs_std = [
                torch.std(ob, dim=0).detach().cpu().numpy() for ob in obs
            ]

            full_traj = [x[:-1] for x in train_x] + \
                [train_x[-1][:-1] + train_y[-1]]
            plot_trajectory_comparison(full_traj, obs_mean, obs_std)

        # update dynamics model
        loss_history = agent.train_dynamics_model(train_x, train_y,
                                                  args.num_dynamics_opt_steps)

        # logging dynamics model metrics
        logger.add_metrics('dynamics_model_loss', n_trial, loss_history)

        # update policy with model-based rollouts
        def policy_training_callback(step, total_cost):
            logger.add_metric('cost', n_trial, step, total_cost)

        agent.train_policy(args.rollout_horizon, args.num_opt_steps, manager,
                           policy_training_callback)

        # evaluation
        eval_costs = evaluate(env, agent, args.num_evaluates, args.render)
        logger.add_metrics('eval_cost', n_trial, eval_costs)

        # save parameters
        agent.save(logger, n_trial)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='pipps-cartpole-swingup-v0')
    parser.add_argument('--noise-k', type=float,
                        default=1.0)  # May also be None
    parser.add_argument('--batch-size', type=int, default=300)
    parser.add_argument('--rbf-centers', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--dynamics-lr', type=float, default=0.1)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--num-random-trials', type=int, default=1)
    parser.add_argument('--num-learned-trials', type=int, default=15)
    parser.add_argument('--num-evaluates', type=int, default=30)
    parser.add_argument('--num-opt-steps', type=int, default=250)
    parser.add_argument('--num-dynamics-opt-steps', type=int, default=100)
    parser.add_argument('--rollout-horizon', type=int, default=30)  # was 25
    parser.add_argument('--init-noise-std', type=float,
                        default=0.02)  # was 0.2
    parser.add_argument('--jitter', type=float, default=1e-6)
    parser.add_argument('--dynamics-model',
                        type=str,
                        choices=['gp'],
                        default='gp')
    parser.add_argument('--method',
                        default='TP',
                        choices=[
                            'TP', 'LR', 'RP', 'RP_no_detach', 'No_proppo',
                            'GR', 'GR_no_detach'
                        ])
    parser.add_argument('--activation',
                        default='sins',
                        choices=['tanh', 'sins'])
    parser.add_argument('--loss', default='Regular', choices=['Regular', 'GS'])
    parser.add_argument('--load-policy', type=str)
    parser.add_argument('--load-dynamics', type=str)
    parser.add_argument('--load-dynamics-data', type=str)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--id', type=str, default='test')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--save-git', action='store_true')
    parser.add_argument('--plot-traj', action='store_true')
    # Set the maximum used number of cpu threads. 0 means that no max is set.
    parser.add_argument('--num-threads', type=int, default=0)
    args = parser.parse_args()
    main(args)
