import argparse, sys, os
from pathlib import Path
import torch

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Common.Utils_model import *
from Common.Buffer import Buffer
from Network.Model_Network import *

torch.autograd.set_detect_anomaly(True)

def hyperparameters():
    parser = argparse.ArgumentParser(description='Soft Actor Critic (SAC) v2 example')
    # save path
    parser.add_argument('--path', default="/home/phb/SBPO/Results/", help='path for save')
    parser.add_argument('--result-index', default="ant_dnn/", help='result to check')

    parser.add_argument("--n_history", default=3, type=int, help="state history stack")

    # estimate a model dynamics
    parser.add_argument('--modelbased-mode', default=True, type=bool,
                        help="you should choose whether basic or model_base")
    parser.add_argument('--develop-mode', '-dm', default='DeepDOB', help="Both, DeepDOB, MRAP")
    parser.add_argument('--ensemble-mode', default="False", type=str2bool,
                        help="you should choose whether using an ensemble ")
    parser.add_argument('--ensemble-size', default=3, type=int, help="ensemble size")
    parser.add_argument('--model-batch-size', default=5, type=int, help="model batch size to use for ensemble")
    parser.add_argument('--net-type', default="DNN", help='all, DNN, BNN, prob')
    parser.add_argument('--model-lr-dnn', default=0.001, type=float)
    parser.add_argument('--model-lr-bnn', default=0.001, type=float)
    parser.add_argument('--model-kl-weight', default=0.05, type=float)
    parser.add_argument('--inv-model-lr-dnn', default=0.001, type=float)
    parser.add_argument('--inv-model-lr-bnn', default=0.001, type=float)
    parser.add_argument('--inv-model-kl-weight', default=1e-6, type=float)
    parser.add_argument('--use-random-buffer', default=True, type=bool, help="add random action to training data")

    parser.add_argument('--render', default=False, type=bool)

    parser.add_argument('--train-step', default=100000000, type=int, help='Maximum training step')
    parser.add_argument('--batch-size', default=256, type=int, help='Mini-batch size')
    parser.add_argument('--buffer-size', default=1000000, type=int, help='Buffer maximum size')
    parser.add_argument('--training-step', default=1, type=int)
    parser.add_argument('--eval-step', default=100000, type=int, help='Frequency in performance evaluation')
    parser.add_argument('--eval-episode', default=5, type=int, help='Number of episodes to perform evaluation')
    parser.add_argument('--prev-result', default='False', type=str2bool, help='if previous result, True')
    parser.add_argument('--prev-result-fname', default='0407_HalfCheetah-v3_esb', help='choose the result to view')
    parser.add_argument('--policynet-name', default='policy_best', help='best, better, current, total')

    args = parser.parse_known_args()

    if len(args) != 1:
        args = args[0]

    return args


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path_policy, env_name, algorithm_name, state_dim, action_dim, max_action, min_action, frameskip, modelbased_mode, ensemble_mode \
        = load_config(args)

    ensemble_mode = args.ensemble_mode

    _, algorithm = get_algorithm_info(algorithm_name, state_dim, action_dim, device)
    args_tester = None

    path = args.path + args.result_index

    if args.use_random_buffer is True:
        buffer4model = Buffer(state_dim=state_dim * args.n_history, action_dim=action_dim * (args.n_history + 1),
                              max_size=args.buffer_size, on_policy=False, device=algorithm.device)
    else:
        buffer4model = None

    buffer4model.load_buffer(path, 'after_full')

    eval_data = DataManager()
    distribution_data = DataManager()

    random_seed = set_seed(-1)
    env, _ = gym_env(env_name, random_seed)
    algorithm.actor.load_state_dict(torch.load(path_policy))

    if args.net_type == 'DNN':
        models = \
            create_models(state_dim, action_dim, frameskip, algorithm,
                          args, args_tester, bnn=False, buffer=buffer4model,
                          ensemble_mode=ensemble_mode)
    elif args.net_type == 'BNN':
        models = \
            create_models(state_dim, action_dim, frameskip, algorithm,
                          args, args_tester, dnn=False, buffer=buffer4model,
                          ensemble_mode=ensemble_mode)
    else:
        models = \
            create_models(state_dim, action_dim, frameskip, algorithm,
                          args, args_tester, buffer=buffer4model,
                          ensemble_mode=ensemble_mode)
    eval_num = 0
    loss_best = None

    for i in range(args.train_step):

        cost_list, mse_list, kl_list = train_alls(args.training_step, models)
        saveData = np.hstack((mse_list, kl_list))
        eval_data.put_data(saveData)

        print("Train | Episode: ", i, ", RMSE: ", np.sqrt(mse_list), ", cost: ", cost_list)

        if i % args.eval_step == 0 and i != 0:

            eval_data.save_data(path, "saved_log/Eval_by" + str(i // args.eval_step))
            eval_data.init_data()

            eval_num += 1
            episode = 0
            reward_list = []
            alive_cnt = 0

            while True:
                local_step = 0
                if episode >= args.eval_episode:
                    break
                episode += 1
                eval_reward = 0
                observation = env.reset()
                done = False

                mem_observation = np.concatenate([observation] * args.n_history)
                mem_next_observation = np.concatenate([observation] * args.n_history)
                mem_action = np.concatenate([np.zeros(action_dim)] * (args.n_history + 1), axis=-1)

                while not done:
                    local_step += 1
                    action = algorithm.eval_action(observation)
                    env_action = denormalize(action, max_action, min_action)

                    next_observation, reward, done, _ = env.step(env_action)

                    mem_observation[state_dim:] = mem_observation[:state_dim * (args.n_history - 1)]
                    mem_observation[:state_dim] = observation
                    mem_next_observation[state_dim:] = mem_observation[
                                                       :state_dim * (args.n_history - 1)]
                    mem_next_observation[:state_dim] = next_observation
                    mem_action[action_dim:] = mem_action[:action_dim * args.n_history]
                    mem_action[:action_dim] = action

                    error_list = eval_models(mem_observation, mem_action, mem_next_observation, models)
                    eval_data.put_data(error_list)

                    if args.render == True:
                        env.render()

                    eval_reward += reward
                    observation = next_observation

                    if local_step == env.spec.max_episode_steps:
                        alive_cnt += 1

                reward_list.append(eval_reward)

            alive_rate = alive_cnt / args.eval_episode

            loss = validate_measure(eval_data.data)
            distribution_data.put_data(np.hstack(loss))
            distribution_data.save_data(path, "saved_log/loss_mean_std")
            eval_loss, mean, std, error_max = loss

            if eval_num == 1:
                loss_best = eval_loss

            loss_with_index = save_models(args, loss_best, eval_loss, path, models)
            if loss_with_index is not None:
                loss_best[loss_with_index[0]] = loss_with_index[1]

            eval_data.save_data(path, "saved_log/Eval_" + str(i // args.eval_step))
            eval_data.init_data()

            print(
                "Eval  | Average Reward: {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f}, alive rate : {:.2f}"
                .format(sum(reward_list) / len(reward_list), max(reward_list), min(reward_list),
                        np.std(reward_list), 100 * alive_rate))

            print("Cost  | ", args.develop_mode, args.net_type, " | ", error_max)
            env.close()


if __name__ == '__main__':
    args = hyperparameters()
    main(args)