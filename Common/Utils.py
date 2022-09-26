import numpy as np
import pandas as pd
import random, math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from IPython.display import clear_output

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


sys.path.append(str(Path('Utils.py').parent.absolute()))  # 절대 경로에 추가


class DataManager:
    def __init__(self, data_name=None):
        self.data = None
        self.xticks = []
        self.data_name = data_name

    def init_data(self):
        self.data = None
        self.xticks = []

    def put_data(self, obs):
        if self.data is None:
            self.data = obs
        else:
            self.data = np.vstack((self.data, obs))

    def get_xticks(self, xtick):
        self.xticks.append(str(xtick))

    def mean_data(self):
        mean_data = np.mean(self.data, axis=0)
        return mean_data

    def plot_data(self, obs, label=None):
        self.put_data(obs)
        if label is None:
            if self.data_name is not None:
                plt.figure(self.data_name)
            plt.plot(self.data, 'o')
        else:
            plt.plot(self.data, label=label)
            plt.legend()


    def save_data(self, path, fname, numpy=False):
        if numpy is False:
            df = pd.DataFrame(self.data)
            df.to_csv(path + fname + ".csv")
        else:
            df = np.array(self.data)
            np.save(path + fname + ".npy", df)

    def plot_fig(self, path):
        clear_output(True)
        f = plt.figure(figsize=(20,5))
        plt.plot(np.arange(len(self.data)), self.data)
        plt.grid(True)
        plt.savefig(path)
        # plt.show()
        f.clf()
        plt.close(f)

    def bar_fig(self, path):
        clear_output(True)
        f = plt.figure(figsize=(20,5))
        x = np.arange(len(self.data))
        plt.bar(x, self.data.flatten())
        plt.xticks(x, self.xticks)
        plt.grid(True)
        plt.savefig(path)
        # plt.show()
        f.clf()
        plt.close(f)

    def plot_variance_fig(self, path, need_xticks=False):
        clear_output(True)
        f = plt.figure(figsize=(20,5))
        mean_val = self.data[:, 0]
        std_val = self.data[:, 1]
        x = range(len(mean_val))
        plt.plot(x, mean_val)
        y1 = np.asarray(mean_val) + np.asarray(std_val)
        y2 = np.asarray(mean_val) - np.asarray(std_val)
        plt.fill_between(x, y1, y2, alpha=0.3)
        plt.grid(True)
        if need_xticks:
            plt.xticks(x, self.xticks)
        plt.savefig(path)
        # plt.show()
        f.clf()
        plt.close(f)

def plot_fig(data, path):
    clear_output(True)
    f = plt.figure(figsize=(20,5))
    x = range(len(data))
    plt.plot(x, data)
    plt.grid(True)
    plt.savefig(path)
    # plt.show()
    f.clf()
    plt.close(f)

def plot_variance_fig(mean_val, std_val, path):
    clear_output(True)
    f = plt.figure(figsize=(20,5))
    x = range(len(mean_val))
    plt.plot(x, mean_val)
    y1 = np.asarray(mean_val) + np.asarray(std_val)
    y2 = np.asarray(mean_val) - np.asarray(std_val)
    plt.fill_between(x, y1, y2, alpha=0.3)
    plt.grid(True)
    plt.savefig(path)
    # plt.show()
    f.clf()
    plt.close(f)

## related to control ##
def quat2mat(quat):
    """ Convert Quaternion to Rotation matrix.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


def rot2rpy(R):
    temp = np.array([0, 0, 1]) @ R
    pitch = math.asin(-temp[0])
    roll = math.asin(temp[1] / math.cos(pitch))
    yaw = math.acos(R[0, 0] / math.cos(pitch))

    return roll, pitch, yaw


def quat2rpy(quat):
    R = quat2mat(quat)
    euler = rot2rpy(R)
    euler = np.array(euler)
    return euler

def normalize(input, act_max, act_min):

    if type(input) is not torch.Tensor:
        normal_mat = np.zeros((len(input), len(input)))
        np.fill_diagonal(normal_mat,  2 / (act_max - act_min))
    else:
        act_max = torch.tensor(act_max, dtype=torch.float).cuda()
        act_min = torch.tensor(act_min, dtype=torch.float).cuda()
        normal_mat = torch.diag(2 / (act_max - act_min))
    normal_bias = (act_max + act_min) / 2
    input = (input - normal_bias) @ normal_mat
    return input

def denormalize(input, act_max, act_min):

    if type(input) is not torch.Tensor:
        denormal_mat = np.zeros((len(input), len(input)))
        np.fill_diagonal(denormal_mat, (act_max - act_min) / 2)
    else:
        act_max = torch.tensor(act_max, dtype=torch.float).cuda()
        act_min = torch.tensor(act_min, dtype=torch.float).cuda()
        denormal_mat = torch.diag((act_max - act_min) / 2)

    denormal_bias = (act_max + act_min) / 2
    input = input @ denormal_mat + denormal_bias

    return input

def inv_softsign(y):
    if type(y) is torch.Tensor:
        y = y.cpu().detach().numpy()

    x = np.where(y >= 0, y/(1-y), y/(1+y))
    return x


def add_noise(val, scale = 0.1):
    val += scale*np.random.normal(size=len(val))
    return val


def add_disturbance(val, step, terminal_time, scale=0.1, frequency=4):

    for i in range(len(val)):
        val[i] += scale*math.sin((frequency*math.pi / terminal_time)*step)

    if scale > 0.01:
        if type(val) is torch.Tensor:
            val += 0.01*torch.normal(mean=torch.zeros_like(val), std=torch.ones_like(val))
        else:
            val += 0.01*np.random.normal(size=len(val))

    return val

## related to saved data ##

def np2str(nump):
    _str = ""
    for element in nump:
        _str += (str(element) + " ")
    return _str

def create_config(algorithm_name, args, env, state_dim, action_dim, max_action, min_action):

    max_action_str = np2str(max_action)
    min_action_str = np2str(min_action)

    with open(args.path + 'config.txt', 'w') as f:
        print("Develop mode:", args.develop_mode, file=f)
        print("Environment:", args.env_name, file=f)
        print("Algorithm:", algorithm_name, file=f)
        print("State dim:", state_dim, file=f)
        print("Action dim:", action_dim, file=f)
        print("Max action:", max_action_str, file=f)
        print("Min action:", min_action_str, file=f)
        print("Step size:", env.env.dt, file=f)
        print("Frame skip:", env.env.frame_skip, file=f)
        print("Save path :", args.path, file=f)
        print("Model based mode:", args.modelbased_mode, file=f)
        print("model lr : {}, model klweight : {}, inv model lr dnn: {}, inv model lr bnn: {}, inv model klweight : {}".
              format(args.model_lr, args.model_kl_weight, args.inv_model_lr_dnn, args.inv_model_lr_bnn, args.inv_model_kl_weight), file=f)
        print("consideration note : ", args.note, file=f)
        print(" ")

def load_config(args):

    if args.prev_result is True:
        path_config = args.path + "storage/" + args.prev_result_fname + "/config.txt"
        path_policy = args.path + "storage/" + args.prev_result_fname + "/saved_net/policy/" + args.policynet_name
    else:
        path_config = args.path + args.result_fname + "config.txt"
        path_policy = args.path + args.result_fname + "saved_net/policy/" + args.policynet_name

    modelbased_mode_cfg = False

    with open(path_config, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            # print(line[:len(line)-1])
            if 'Environment:' in line:
                env_name_cfg = line[line.index(':')+2:len(line)-1]
            if 'Algorithm:' in line:
                algorithm_cfg = line[line.index(':')+2:len(line)-1]
            if 'State dim:' in line:
                state_dim_cfg = int(line[line.index(':')+2:len(line)-1])
            if 'Action dim:' in line:
                action_dim_cfg = int(line[line.index(':')+2:len(line)-1])
            if 'Max action:' in line:
                max_action_cfg = np.fromstring(line[line.index(':')+2:len(line)-1], dtype=float, sep=" ")
            if 'Min action:' in line:
                min_action_cfg = np.fromstring(line[line.index(':')+2:len(line)-1], dtype=float, sep=" ")
            if 'Frame skip:' in line:
                frame_skip_cfg = int(line[line.index(':')+2:len(line)-1])
            if 'Model based mode:' in line:
                modelbased_mode_cfg = (line[line.index(':')+2:len(line)-1] == 'True')

    return path_policy, env_name_cfg, algorithm_cfg, state_dim_cfg, action_dim_cfg, max_action_cfg, min_action_cfg, \
           frame_skip_cfg, modelbased_mode_cfg


def get_algorithm_info(algorithm_name, state_dim, action_dim, device):

    # print(algorithm_name)
    # print('SAC_v2')
    # print(algorithm_name == 'SAC_v2')

    if algorithm_name == 'SAC_v3':
        from run_SACv3 import hyperparameters
        from Algorithm.SAC_v3 import SAC_v3
        _args = hyperparameters()
        _algorithm = SAC_v3(state_dim, action_dim, device, _args)
    else:
        raise Exception("check the name of algorithm")
    return _args, _algorithm

def save_policy(policy, score_best, score_now, alive_rate, path):

    if score_now > score_best:
        torch.save(policy.state_dict(), path + "/policy_better")
        if alive_rate > 0.8:
            torch.save(policy.state_dict(), path + "/policy_best")
        return score_now

    torch.save(policy.state_dict(), path + "/policy_current")

def load_model(network, path, fname):
    if "bnn" in fname:
        model_tmp = torch.load(path +'/' + fname)
        for key in model_tmp.copy().keys():
            if 'eps' in key:
                del (model_tmp[key])
        network.load_state_dict(model_tmp)
    else:
        network.load_state_dict(torch.load(path +'/' + fname))
    network.eval()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')

## related to gym

def set_seed(random_seed):
    if random_seed <= 0:
        random_seed = np.random.randint(1, 9999)
    else:
        random_seed = random_seed

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    random.seed(random_seed)

    return random_seed

def gym_env(env_name, random_seed):
    import gym
    # openai gym
    env = gym.make(env_name)
    env.seed(random_seed)
    env.action_space.seed(random_seed)

    test_env = gym.make(env_name)
    test_env.seed(random_seed)
    test_env.action_space.seed(random_seed)

    return env, test_env

def suite_env(*arg):
    import robosuite as suite

    # robosuite
    env = suite.make(env_name=arg[0],
                     robots=arg[1],
                     has_renderer=arg[2],
                     has_offscreen_renderer=arg[3],
                     use_camera_obs=arg[4],
                     reward_shaping=True)

    test_env = suite.make(env_name=arg[0],
                     robots=arg[1],
                     has_renderer=arg[2],
                     has_offscreen_renderer=arg[3],
                     use_camera_obs=arg[4],
                     reward_shaping=True)

    return env, test_env

def obs_to_state(env, obs):
    state = None
    for key in env.active_observables:
        if state is None:
            state = obs[key]
        else:
            state = np.hstack((state, obs[key]))
    return state

def soft_update(network, target_network, tau):
    for param, target_param in zip(network.parameters(), target_network.parameters()):
        target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers.
        Reference: https://github.com/MishaLaskin/rad/blob/master/curl_sac.py"""

    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

def copy_weight(network, target_network):
    target_network.load_state_dict(network.state_dict())

def share_parameters(adamoptim):
    ''' share parameters of Adamoptimizers for multiprocessing '''
    for group in adamoptim.param_groups:
        for p in group['params']:
            state = adamoptim.state[p]
            # initialize: have to initialize here, or else cannot find
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(p.data)
            state['exp_avg_sq'] = torch.zeros_like(p.data)

            # share in memory
            state['exp_avg'].share_memory_()
            state['exp_avg_sq'].share_memory_()
