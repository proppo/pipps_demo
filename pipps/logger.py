import os
import re
import json
import cv2
import torch
import pickle
import subprocess

from datetime import datetime


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', \
                                    'HEAD']).decode('ascii').strip()


def prepare_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_latest_trial(dir_path):
    file_names = os.listdir(dir_path)
    latest_trial = -1
    for file_name in file_names:
        match = re.match(r'policy_(\d+).pt', file_name)
        if match:
            latest_trial = max(int(match.group(1)), latest_trial)
    return latest_trial


def save_video(images, path):
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(path, fourcc, 10, (width, height))
    for image in images:
        writer.write(image)
    writer.release()


class Logger:

    def __init__(self, dir_path):
        self.dir_path = dir_path
        prepare_directory(dir_path)

    def add_params(self, params):
        with open(self._get_path('params.json'), 'w') as f:
            f.write(json.dumps(params))
        for k, v in params.items():
            print('{}={}'.format(k, v))

    def save_githash(self):
        with open(self._get_path('githash.txt'), 'w') as f:
            f.write('git-hash={}'.format(get_git_revision_hash()))

    def add_metric(self, name, n_trial, n_iter, value):
        with open(self._get_path(name, '.csv'), 'a') as f:
            print('{},{},{}'.format(n_trial, n_iter, value), file=f)
        message = '{}: trial={} iter={} value={}'
        print(message.format(name, n_trial, n_iter, value))

    def add_metrics(self, name, n_trial, values):
        for n_iter, value in enumerate(values):
            self.add_metric(name, n_trial, n_iter, value)

    def _get_path(self, name, suffix=None):
        path = os.path.join(self.dir_path, name)
        if suffix:
            path = path + suffix
        return path

    def add_video(self, name, n_trial, n_iter, images):
        video_dir_path = os.path.join(self.dir_path, name)
        prepare_directory(video_dir_path)
        video_file_name = '{}_{}.avi'.format(n_trial, n_iter)
        video_file_path = os.path.join(video_dir_path, video_file_name)
        save_video(images, video_file_path)

    def add_videos(self, name, n_trial, images_list):
        for n_iter, images in enumerate(images_list):
            self.add_video(name, n_trial, n_iter, images)

    def add_model(self, name, n_trial, model):
        name += '_{}'.format(n_trial)
        model_path = self._get_path(name, '.pt')
        torch.save(model.state_dict(), model_path)
        print('Parameters saved to ', model_path)

    def add_data(self, name, n_trial, data):
        name += '_{}'.format(n_trial)
        data_path = self._get_path(name, '.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        print('Pickled data saved to ', data_path)


def prepare_logger(dir_path, name, timestamp=False):
    if timestamp:
        date = '_' + datetime.now().strftime('%Y%m%d%H%M%S')
    else:
        date = ''
    dir_path = os.path.join(dir_path, name + date)
    return Logger(dir_path)
