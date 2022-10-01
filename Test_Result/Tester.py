import argparse, sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))   # 절대 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Algorithm.SAC_v3 import *
from Common.Utils import *
from Common.Buffer import *
from Common.logger import *

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

    algorithm = SAC_v3(state_dim, action_dim, replay_buffer, args, device)
    np.random.seed(77)

    eval_reward = DataManager()
    eval_success = DataManager()

    log_dir = load_log_directories(args.result_name)
    load_model(sac_trainer.policy_net, log_dir["policy"], "policy_best")
    if args.model_on:
        load_model(sac_trainer.inv_model_net, log_dir[args.net_type], "better_" + args.net_type)
        sac_trainer.inv_model_net.evaluates()
        print('Sparsification ratio: %.3f%%' % (100. * nn_ard.get_dropped_params_ratio(sac_trainer.inv_model_net)))
    env = Sim2RealEnv(args=args)

    result_txt = open(log_dir["test"] + "/test_result_%s" % time.strftime(
        "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.add_to + ".txt", 'w')

    if args.add_to == "action":
        min_dist = args.min_dist_action
        max_dist = args.max_dist_action
    else:
        min_dist = args.min_dist_state
        max_dist = args.max_dist_state

    for dist_scale in np.round(np.linspace(min_dist, max_dist, args.num_dist + 1), 3):
        if args.add_to == "action":
            env.dist_scale = dist_scale
            print("disturbance scale: ", dist_scale * 100, " percent of max thrust", file=result_txt)
            print("disturbance scale: ", dist_scale * 100, " percent of max thrust")
            eval_reward.get_xticks(np.round(dist_scale * 100, 3))
            eval_success.get_xticks(np.round(dist_scale * 100, 3))
        else:
            print("standard deviation of state noise: ", dist_scale, file=result_txt)
            print("standard deviation of state noise: ", dist_scale)
            eval_reward.get_xticks(dist_scale)
            eval_success.get_xticks(dist_scale)

        success_rate = 0
        reward_list = []
        suc_reward = 0

        for eps in range(args.test_eps):
            state = env.reset()
            episode_reward = 0
            p = state["position_error_obs"]
            v = state["velocity_error_obs"]
            r = state["rotation_obs"]
            w = state["angular_velocity_error_obs"]
            a = state["action_obs"]
            pos = p[:3]
            vel = -v[:3]
            rpy = np.array([math.atan2(r[0], r[1]), math.atan2(r[2], r[3]), math.atan2(r[4], r[5])])
            angvel = -w[:3]
            policy = a[:4]
            force = np.zeros(4)
            step = 0
            episode_model_error = []
            dist = np.zeros(env.action_dim)
            dist_before = np.zeros(env.action_dim)

            for step in range(args.episode_length):
                network_state = np.concatenate([p, v, r, w])
                action = sac_trainer.policy_net.get_action(network_state, deterministic=True)

                if args.model_on:
                    action_dob = action - dist
                    next_state, reward, done, success, f = env.step(action_dob)

                    if args.add_to == "state":
                        for k in next_state.keys():
                            next_state[k] = np.random.normal(next_state[k], dist_scale)

                    network_states = get_model_net_input(env, state, next_state=next_state, ver=args.develop_version)

                    if args.develop_version == 1:
                        network_state, prev_network_action, next_network_state = network_states
                    else:
                        _, prev_network_action, next_network_state = network_states

                    action_hat = sac_trainer.inv_model_net(network_state, prev_network_action,
                                                           next_network_state).detach().cpu().numpy()[0]
                    dist = action_hat - action
                    dist = np.clip(dist, -1.0, 1.0)
                    episode_model_error.append(np.sqrt(np.mean(dist ** 2)))
                else:
                    next_state, reward, done, success, f = env.step(action)
                    if args.add_to == "state":
                        for k in next_state.keys():
                            next_state[k] = np.random.normal(next_state[k], dist_scale)

                episode_reward += reward
                state = next_state
                if args.render:
                    env.render()

                p = state["position_error_obs"]
                v = state["velocity_error_obs"]
                r = state["rotation_obs"]
                w = state["angular_velocity_error_obs"]
                a = state["action_obs"]

                pos = np.vstack((pos, p[:3]))
                vel = np.vstack((vel, -v[:3]))
                rpy = np.vstack(
                    (rpy, np.array([math.atan2(r[0], r[1]), math.atan2(r[2], r[3]), math.atan2(r[4], r[5])])))
                angvel = np.vstack((angvel, -w[:3]))
                policy = np.vstack((policy, a[:4]))
                force = np.vstack((force, f))

                if done or success:
                    break

            # when you want to know specific data
            # eval_plot(step, pos, vel, rpy, angvel, policy, force)

            print('Episode: ', eps, '| Episode Reward: ', episode_reward, '| Episode Model error: ',
                  np.mean(episode_model_error))
            reward_list.append(episode_reward)
            if episode_reward > 300:
                suc_reward += episode_reward
                success_rate += 1.

        if success_rate != 0:
            suc_reward /= success_rate
        else:
            suc_reward = 0.

        success_rate /= args.test_eps
        avg_reward = sum(reward_list) / args.test_eps
        eval_reward.put_data((np.mean(reward_list), np.std(reward_list)))
        eval_success.put_data(success_rate * 100)
        print('Success rate: ', success_rate * 100, '| Average Reward: ', avg_reward, '| Success Reward: ', suc_reward,
              file=result_txt)
        print('Success rate: ', success_rate * 100, '| Average Reward: ', avg_reward, '| Success Reward: ', suc_reward)

    eval_reward.plot_variance_fig(log_dir["test"] + "/reward_%s" % time.strftime(
        "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.add_to, need_xticks=True)
    eval_reward.save_data(log_dir["test"], "/reward_%s" % time.strftime(
        "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.add_to, numpy=True)
    eval_success.bar_fig(log_dir["test"] + "/success_rate_%s" % time.strftime(
        "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.add_to)
    eval_success.save_data(log_dir["test"], "/success_rate_%s" % time.strftime(
        "%m%d-%H%M_") + args.develop_mode + "_" + args.net_type + "_" + args.add_to, numpy=True)
    result_txt.close()

if __name__ == '__main__':
    args_tester = hyperparameters()
    main(args_tester)