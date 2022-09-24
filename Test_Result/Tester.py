import sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))   # 절대 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Example import *
from Common.Utils import *

def hyperparameters(result_fname="0503_Hopper",
                    num_test=5,
                    noise_scale=0.0,
                    disturbance_scale=0.0,
                    disturbance_freq=16,
                    add_to='action',
                    policy_name="policy_best",
                    model_name="dnn_better"
                    ):

    parser = argparse.ArgumentParser(description='Tester of algorithms')

    # related to development
    parser.add_argument('--test-on', default='False', type=str2bool, help="You must turn on when you test")
    parser.add_argument('--develop-mode', '-dm', default='MRAP', help="Basic, DeepDOB, MRAP")
    parser.add_argument('--frameskip_inner', default=1, type=int, help='frame skip in inner loop ')

    # environment
    parser.add_argument('--render', default='False', type=str2bool)
    parser.add_argument('--test-episode', default=num_test, type=int, help='Number of episodes to perform evaluation')

    # result to watch
    parser.add_argument('--path', default="/home/phb/SBPO/Results/", help='path for save')
    parser.add_argument('--result-fname', default=result_fname + "/", help='result to check')
    parser.add_argument('--prev-result', default='False', type=str2bool, help='if previous result, True')
    parser.add_argument('--prev-result-fname', default=result_fname, help='choose the result to view')
    parser.add_argument('--modelnet-name', default=model_name, help='modelDNN_better, modelBNN_better')
    parser.add_argument('--policynet-name', default=policy_name, help='best, better, current, total')
    parser.add_argument('--ensemble-size', default=3, type=int, help="ensemble size")

    # setting real world
    parser.add_argument('--add_noise', default='False', type=str2bool, help="if True, add noise to action")
    parser.add_argument('--noise_to', default=add_to, help="state, action")
    parser.add_argument('--noise_scale', default=noise_scale, type=float, help='white noise having the noise scale')

    parser.add_argument('--add_disturbance', default='True', type=str2bool, help="if True, add disturbance to action")
    parser.add_argument('--disturbance_to', default=add_to, help="state, action")
    parser.add_argument('--disturbance_scale', default=disturbance_scale, type=float, help='choose disturbance scale')
    parser.add_argument('--disturbance_frequency', default=disturbance_freq, type=list, help='choose disturbance frequency')

    # Etc
    parser.add_argument('--cpu-only', default='False', type=str2bool, help='force to use cpu only')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')

    args = parser.parse_known_args()

    if len(args) != 1:
        args = args[0]

    return args

def main(args_tester):

    if args_tester.test_on is False:
        raise Exception(" You must turn on args_tester.test_on if you wanna test ")

    if args_tester.cpu_only == True:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print("Device:", device)

    path_policy, env_name, algorithm_name, state_dim, action_dim, max_action, min_action, frameskip, modelbased_mode\
        = load_config(args_tester)
    print(path_policy)
    print("result_name:", args_tester.result_fname)

    args, algorithm = get_algorithm_info(algorithm_name, state_dim, action_dim, device)

    random_seed = set_seed(args.random_seed)
    env, test_env = gym_env(env_name, random_seed)
    algorithm.actor.load_state_dict(torch.load(path_policy))
    algorithm.dynamics.load_state_dict(torch.load(args_tester.path
                                                  + args_tester.result_fname
                                                  + "saved_net/model/"
                                                  + args_tester.modelnet_name))
    algorithm.pid.load_state_dict(torch.load(args_tester.path
                                             + args_tester.result_fname
                                             + "saved_net/policy/pid_best"
                                             ))

    print('develop-mode:', args_tester.develop_mode)
    print('noise scale:', args_tester.noise_scale, 'disturbance scale:', args_tester.disturbance_scale)
    print('policy_name:', args_tester.policynet_name)
    print('modelnet_name:', args_tester.modelnet_name)

    if args_tester.develop_mode == 'Basic':
        trainer = Basic_trainer(
            env, test_env, algorithm,
            state_dim, action_dim, max_action, min_action,
            args, args_tester)
    else:
        trainer = Model_trainer(
            env, test_env, algorithm,
            state_dim, action_dim, max_action, min_action,
            args, args_tester)
    reward_avg, reward_max, reward_min, reward_std, alive_rate = trainer.test()

    return reward_avg, reward_max, reward_min, reward_std, alive_rate

if __name__ == '__main__':
    args_tester = hyperparameters()
    main(args_tester)