import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def plot_trajectory_comparison(actual_obs_history, pred_obs_history,
                               pred_std_history):
    actual_obs_history = np.array(actual_obs_history)
    pred_obs_history = np.array(pred_obs_history)
    pred_std_history = np.array(pred_std_history)

    steps = np.arange(pred_obs_history.shape[0])
    lower = pred_obs_history - 2 * pred_std_history
    upper = pred_obs_history + 2 * pred_std_history

    # plot
    f = plt.figure(figsize=(15, 15), constrained_layout=True)
    ax1 = f.add_subplot(4, 1, 1)
    plt.title('Position', fontsize=14)
    plt.plot(actual_obs_history[:, 0], label='Actual trajectory')
    plt.plot(pred_obs_history[:, 0], label='Predicted trajectory')
    plt.fill_between(steps,
                     lower[:, 0],
                     upper[:, 0],
                     alpha=0.5,
                     color='orange')
    ax1.xaxis.set_tick_params(labelsize=12)
    ax1.yaxis.set_tick_params(labelsize=12)

    ax2 = f.add_subplot(4, 1, 2)
    plt.title('Angle', fontsize=14)
    plt.plot(actual_obs_history[:, 1], label='Actual trajectory')
    plt.plot(pred_obs_history[:, 1], label='Predicted trajectory')
    plt.fill_between(steps,
                     lower[:, 1],
                     upper[:, 1],
                     alpha=0.5,
                     color='orange')
    ax2.xaxis.set_tick_params(labelsize=12)
    ax2.yaxis.set_tick_params(labelsize=12)

    ax3 = f.add_subplot(4, 1, 3)
    plt.title('Velocity (position)', fontsize=14)
    plt.plot(actual_obs_history[:, 2], label='Actual trajectory')
    plt.plot(pred_obs_history[:, 2], label='Predicted trajectory')
    plt.fill_between(steps,
                     lower[:, 2],
                     upper[:, 2],
                     alpha=0.5,
                     color='orange')
    ax3.xaxis.set_tick_params(labelsize=12)
    ax3.yaxis.set_tick_params(labelsize=12)

    ax4 = f.add_subplot(4, 1, 4)
    plt.title('Velocity (angle)', fontsize=14)
    plt.plot(actual_obs_history[:, 3], label='Actual trajectory')
    plt.plot(pred_obs_history[:, 3], label='Predicted trajectory')
    plt.fill_between(steps,
                     lower[:, 3],
                     upper[:, 3],
                     alpha=0.5,
                     color='orange')
    ax4.xaxis.set_tick_params(labelsize=12)
    ax4.yaxis.set_tick_params(labelsize=12)

    plt.legend(fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    f.subplots_adjust(hspace=0.85)
    plt.show()
