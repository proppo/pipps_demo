import os
import json

from pipps.logger import get_latest_trial


def resume(dir_path, args):
    trial = get_latest_trial(dir_path)
    if trial < 0:
        return
    policy_path = os.path.join(dir_path, 'policy_%d.pt' % trial)
    args.load_policy = policy_path
    dynamics_path = os.path.join(dir_path, 'model_%d.pt' % trial)
    args.load_dynamics = dynamics_path
    dynamics_data_path = os.path.join(dir_path, 'dynamics_data_%d.pkl' % trial)
    args.load_dynamics_data = dynamics_data_path

    # load parameters
    with open(os.path.join(dir_path, 'params.json'), 'r') as f:
        params = json.loads(f.read())
    args.dynamics_model = params['dynamics_model']

    return trial
