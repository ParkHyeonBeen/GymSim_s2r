import gym

from Common.Utils_model import *
from Network.Model_Network import *
from Common.logger import *
import torch_ard as nn_ard

eval_data = DataManager()

def model_trainer(id, algorithm, rewards_queue, replay_buffer, model_path, args, log_queue, startTime=None):

    setup_worker_logging(rank=0, log_queue=log_queue)

    # Configure environments to train

    env = gym.make(args.env_name)
    max_action = env.action_space.high
    min_action = env.action_space.low
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    eps = 0
    eval_freq = args.eval_step
    best_score = None
    best_error = None
    best_error_bnn = None

    """
    fail score for the environments which don't have uncertain criterion about fail
    fail score = maximum score * 0.2

    Half cheetah    : 1183.44
    Ant             : 647.3822
    """
    if args.env_name == "HalfCheetah-v4":
        fail_score = 2 * 1183.44
    elif args.env_name == "Ant-v4":
        fail_score = 2 * 647.3822
    else:
        fail_score = 0.

    while algorithm.worker_step < args.max_step:

        if args.use_prev_policy is True:
            load_model(algorithm.actor, "./Etc/policys/" + args.env_name, "policy_best")
        else:
            if args.model_train_start_step <= algorithm.worker_step < args.model_train_start_step + env.spec.max_episode_steps:
                load_model(algorithm.actor, model_path["policy"], "policy_better")

        episode_reward = 0
        observation = env.reset()
        step = 0

        mem_observation = np.concatenate([observation] * args.n_history)
        mem_action = np.concatenate([np.zeros(action_dim)] * args.n_history)

        for step in range(env.spec.max_episode_steps):
            if args.use_prev_policy is True:
                action = algorithm.eval_action(observation)
            else:
                if algorithm.worker_step > args.training_start * args.num_worker:
                    if args.model_train_start_step <= algorithm.worker_step:
                        action = algorithm.eval_action(observation)
                    else:
                        action = algorithm.get_action(observation)
                else:
                    env_action = env.action_space.sample()
                    action = normalize(env_action, max_action, min_action)

            env_action = denormalize(action, max_action, min_action)

            next_observation, reward, done, info = env.step(env_action)

            mem_observation[state_dim:] = mem_observation[:state_dim * (args.n_history - 1)]
            mem_observation[:state_dim] = observation
            mem_action[action_dim:] = mem_action[:action_dim * (args.n_history - 1)]
            mem_action[:action_dim] = action

            if algorithm.worker_step + 1 == env.spec.max_episode_steps:
                real_done = 0.
            else:
                real_done = float(done)

            episode_reward += reward
            replay_buffer.add(mem_observation, mem_action, reward, next_observation, real_done)

            observation = next_observation

            # Update networks per step
            if algorithm.worker_step > args.training_start * args.num_worker:
                if id < args.num_update_worker:
                    for i in range(args.update_iter):
                        try:
                            _ = algorithm.update(algorithm.worker_step)
                            algorithm.update_step += torch.tensor([1])
                        except:
                            logging.error(traceback.format_exc())
                            # pass
            if done:
                break

        if step < 3:
            continue

        eps += 1
        algorithm.worker_step += torch.tensor([step + 1])
        algorithm.eps += torch.tensor([1])

        s = int(time.time() - startTime)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        logging.info(
            'Global Episode: {0} | Episode Reward: {1:.2f} | Local step: {7} | Global worker step: {2} | Update step: {8} | Worker id: {3} | Elapsed time: {4:02d}:{5:02d}:{6:02d}'.format(
                algorithm.eps.tolist()[0], episode_reward, algorithm.worker_step.tolist()[0], id, h, m,
                s, step + 1, algorithm.update_step.tolist()[0]))

        # Evaluation while training
        if id == 0 and algorithm.worker_step.tolist()[0] > eval_freq:
            print('--------------Evaluation start--------------')
            try:
                eval_freq += args.eval_step
                episode_rewards = []
                episodes_model_error = []
                success_cnt = 0
                for eval_step in range(args.eval_episode):
                    episode_reward = 0
                    episode_model_error = []
                    observation = env.reset()

                    mem_observation = np.concatenate([observation] * args.n_history)
                    mem_action = np.concatenate([np.zeros(action_dim)] * (args.n_history-1))

                    for step in range(env.spec.max_episode_steps):
                        action = algorithm.eval_action(observation)
                        env_action = denormalize(action, max_action, min_action)
                        next_observation, reward, done, _ = env.step(env_action)

                        if (args.develop_mode == "imn" and algorithm.worker_step.tolist()[
                            0] > args.model_train_start_step) or args.use_prev_policy is True:
                            algorithm.imn.evaluates()
                            action_hat = algorithm.imn(mem_observation,
                                                       mem_action,
                                                       next_observation).detach().cpu().numpy()
                            episode_model_error.append(np.sqrt(np.mean((action_hat - action) ** 2)))

                            mem_observation[state_dim:] = mem_observation[:state_dim * (args.n_history-1)]
                            mem_observation[:state_dim] = observation
                            mem_action[action_dim:] = mem_action[:action_dim * (args.n_history - 2)]
                            mem_action[:action_dim] = action

                        # env.render()
                        observation = next_observation
                        episode_reward += reward

                        if done:
                            break

                    if step + 1 == env.spec.max_episode_steps and episode_reward > fail_score:
                        success_cnt += 1

                    episode_rewards.append(episode_reward)
                    episodes_model_error.append(episode_model_error)
                avg_reward = np.mean(episode_rewards)

                if best_score is None:
                    best_score = avg_reward

                success_rate = success_cnt / args.eval_episode
                best_score_tmp = save_policy(algorithm.actor, best_score, avg_reward, success_rate,
                                             model_path['policy'])
                if best_score_tmp is not None:
                    best_score = best_score_tmp

                if (args.develop_mode == "imn" and algorithm.worker_step.tolist()[0] > args.model_train_start_step)\
                    or args.use_prev_policy is True:
                    eval_error = np.mean([np.mean(episode_errors) for episode_errors in episodes_model_error],
                                         keepdims=True)
                    eval_data.put_data(eval_error)

                    if best_error is None:
                        best_error = eval_error[0]

                    best_error_tmp = save_model(algorithm.imn, best_error, eval_error[0],
                                                model_path[args.net_type])
                    if best_error_tmp is not None:
                        best_error = best_error_tmp

                    if args.net_type == "bnn":
                        if best_error_bnn is None:
                            best_error_bnn = eval_error[0]/nn_ard.get_dropped_params_ratio(algorithm.imn)

                        now_error = eval_error[0]/nn_ard.get_dropped_params_ratio(algorithm.imn)

                        best_error_bnn_tmp = save_model(algorithm.imn, best_error_bnn, now_error,
                                                    model_path[args.net_type], ard=True)
                        if best_error_bnn_tmp is not None:
                            best_error_bnn = best_error_bnn_tmp

                    eval_data.plot_fig(model_path['train'] + "/model_error.png")

                rewards = [avg_reward, algorithm.worker_step.tolist()[0]]
                rewards_queue.put(rewards)
                logging.error(
                    'Episode Reward: {1:.2f} | Success cnt: {4} | Local Step: {3} | Global Episode: {0} | Global Worker Step: {2}| Sparsification ratio : {5:.3f}'
                    .format(algorithm.eps.tolist()[0], avg_reward, algorithm.worker_step.tolist()[0],
                            step + 1, success_cnt, 100*nn_ard.get_dropped_params_ratio(algorithm.imn)))
            except:
                logging.error(traceback.format_exc())
    rewards_queue.put(None)