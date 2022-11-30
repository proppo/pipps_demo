import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse


def main(args):
    # load either the cost.csv, eval_cost.csv or dynamics_model_loss.csv files
    df = pd.read_csv(args.path, names=['Episode', 'Iteration', 'Loss'])
    sns.set(font_scale=1.5, rc={'text.usetex': True})
    sns.lineplot(data=df, x='Episode', y='Loss')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    main(args)
