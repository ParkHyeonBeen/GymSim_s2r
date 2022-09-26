import argparse, sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Algorithm.SAC_v3 import *
from Trainer import *
from Common.Utils import *
from Common.Buffer import *
from Common.logger import *

# fork, forkserver for Linux
# spawn for Windows 10
torch.multiprocessing.set_start_method('spawn', force=True) # critical for make multiprocessing work
import torch.multiprocessing as mp
from multiprocessing import Process, Manager, freeze_support
from multiprocessing.managers import BaseManager

torch.autograd.set_detect_anomaly(True)

def hyperparameters(env_name="Hopper-v4"):
    parser = argparse.ArgumentParser(description='Soft Actor Critic (SAC) v3 example')
    parser.add_argument('--base_path', default="/home/phb/ETRI/GymSim_s2r/", help='base path of the current project')

    # note in txt
    parser.add_argument('--note',
                        default="change bnn std 0.1 --> 0.01",
                        type=str, help='note about what to change')

    #multi processing
    parser.add_argument("--num_worker", '-nw', default=1, type=int, help="The number of agents for collect data.")
    parser.add_argument("--num_update_worker", '-nuw', default=1, type=int, help="The number of agents for update networks.")
    parser.add_argument('--eval_step', '-es', default=10000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--max_step', '-ms', default=6e6, type=int, help='Maximum training step')
    parser.add_argument('--model_train_start_step', '-mtss', default=3e6, type=int)
    parser.add_argument('--reg_weight', '-rw', default=1.0e-6, type=int)

    # estimate a model dynamics
    parser.add_argument('--develop-mode', '-dm', default='imn', help="none, mn, imn")
    parser.add_argument('--net-type', default="bnn", help='dnn, bnn')
    parser.add_argument('--model-hidden-dim', default=256, type=int)
    parser.add_argument('--model-lr', default=3e-4, type=float)
    parser.add_argument('--inv-model-lr-dnn', default=3e-4, type=float)
    parser.add_argument('--kl-weight', default=0.05, type=float)
    parser.add_argument("--eps-p", default=1e-1, type=float, help="Standard deviation for spatial smoothness")
    parser.add_argument("--lambda-t", default=1e-1, type=float, help="Temporal smoothness for policy loss")
    parser.add_argument("--lambda-s", default=5e-1, type=float, help="Spatial smoothness for policy loss")
    parser.add_argument("--n-history", default=3, type=int, help="state history stack")
    parser.add_argument('--training-start', default=1000, type=int, help='First step to start training')
    parser.add_argument('--eval-episode', default=10, type=int, help='Number of episodes to perform evaluation')

    #environment
    parser.add_argument('--domain-type', default='gym', type=str, help='gym')
    parser.add_argument('--env-name', '-en', default=env_name, help='Pendulum-v0, MountainCarContinuous-v0, Door')
    parser.add_argument('--robots', default='Panda', help='if domain type is suite, choose the robots')
    parser.add_argument('--discrete', default=False, type=bool, help='Always Continuous')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--eval', default=True, type=bool, help='whether to perform evaluation')
    parser.add_argument('--random-seed', default=-1, type=int, help='Random seed setting')

    #sac
    parser.add_argument('--batch-size', default=256, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=1000000, type=int, help='Buffer maximum size')
    parser.add_argument('--update_iter', default=1, type=int)
    parser.add_argument('--train-alpha', default=False, type=bool)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--actor-lr', default=0.001, type=float)
    parser.add_argument('--critic-lr', default=0.001, type=float)
    parser.add_argument('--alpha-lr', default=0.0001, type=float)
    parser.add_argument('--encoder-lr', default=0.001, type=float)
    parser.add_argument('--tau', default=0.01, type=float)
    parser.add_argument('--critic-update', default=1, type=int)
    parser.add_argument('--hidden-dim', default=(256, 256), help='hidden dimension of network')
    parser.add_argument('--log_std_min', default=-10, type=int, help='For squashed gaussian actor')
    parser.add_argument('--log_std_max', default=2, type=int, help='For squashed gaussian actor')

    parser.add_argument('--cpu-only', default=False, type=bool, help='force to use cpu only')

    args = parser.parse_known_args()

    if len(args) != 1:
        args = args[0]

    return args

def main(args):
    if args.cpu_only == True:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # random seed setting
    random_seed = set_seed(args.random_seed)
    print("Develop mode:", args.develop_mode)

    #env setting
    env, test_env = gym_env(args.env_name, random_seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    BaseManager.register('Buffer', Buffer)
    manager = BaseManager()
    manager.start()
    replay_buffer = manager.Buffer(state_dim=state_dim*args.n_history,
                                   action_dim=action_dim*args.n_history,
                                   next_state_dim=state_dim,
                                   args=args,
                                   max_size=args.buffer_size,
                                   on_policy=False,
                                   device=device)

    algorithm = SAC_v3(state_dim, action_dim, replay_buffer, args, device)

    log_dir = create_log_directories("./results", args.env_name)
    update_log_file = log_dir['train'] + "/step.log"
    config_log_file = log_dir['base'] + "/config.log"
    evaluation_log_file = log_dir['train'] + "/evaluation.log"
    config_logger = create_config_logger(args, file=config_log_file)
    startTime = time.time()

    log_queue = setup_primary_logging(update_log_file, evaluation_log_file)

    # share the global parameters in multiprocessing
    algorithm.critic1.share_memory()
    algorithm.critic2.share_memory()
    algorithm.target_critic1.share_memory()
    algorithm.target_critic2.share_memory()
    algorithm.actor.share_memory()
    algorithm.log_alpha.share_memory_()  # variable
    algorithm.worker_step.share_memory_()
    algorithm.update_step.share_memory_()
    algorithm.eps.share_memory_()
    share_parameters(algorithm.actor_optimizer)
    share_parameters(algorithm.critic1_optimizer)
    share_parameters(algorithm.critic2_optimizer)
    share_parameters(algorithm.alpha_optimizer)

    algorithm.imn.share_memory()
    share_parameters(algorithm.imn_optimizer)

    rewards_queue = mp.Queue()

    processes = []
    rewards = []
    eval_step = []
    eval_num = 1

    for i in range(args.num_worker):
        process = Process(target=model_trainer, args=(
            i, algorithm, rewards_queue, replay_buffer, log_dir, args, log_queue,
            startTime))  # the args contain shared and not shared
        process.daemon = True  # all processes closed when the main stops
        processes.append(process)

    for p in processes:
        p.start()
        time.sleep(5)

    while True:  # keep geting the episode reward from the queue
        r = rewards_queue.get()
        if r is not None:
            rewards.append(r[0])
            eval_step.append(r[1])
            del r
        else:
            break
        if len(rewards) > 0:
            plot_fig(rewards, log_dir['train'] + "/reward.png")

    [p.join() for p in processes]  # finished at the same time


if __name__ == '__main__':
    # freeze_support()
    #
    # # env_list = ["Walker2d-v4", "Ant-v4", "Humanoid-v4", "Hopper-v4", "InvertedDoublePendulum-v4", "HumanoidStandup-v4"]
    # env_list = ["HalfCheetah-v4", "Hopper-v4", "Humanoid-v4", "Walker2d-v4", "Ant-v4", "HumanoidStandup-v4"]
    #
    # for env in env_list:
    #     args = hyperparameters(env)
    #     main(args)
    args = hyperparameters()
    main(args)

