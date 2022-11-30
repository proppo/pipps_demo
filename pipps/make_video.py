import argparse
import pickle
import gym
import pipps.envs

from pipps.logger import save_video


def main(args):
    env = gym.make(args.env)

    with open(args.path, 'rb') as f:
        states = pickle.load(f)

    # playback from physics states
    images = []
    env.reset()
    for state in states:
        env.set_physics_state(state)
        images.append(env.render('rgb_array'))

    save_video(images, args.target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--path', type=str)
    parser.add_argument('--target', type=str, default='video.avi')
    args = parser.parse_args()
    main(args)
