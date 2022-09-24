import math, random, time

import numpy as np

from Common.Utils import *
from Common.Buffer import Buffer
from Common.Utils_model import *

class Basic_trainer():
    def __init__(self, env, test_env, algorithm,
                 state_dim, action_dim,
                 max_action, min_action,
                 args, args_tester=None):

        self.args = args
        self.args_tester = args_tester

        self.n_history = self.args.n_history
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.domain_type = self.args.domain_type
        self.env_name = self.args.env_name
        self.env = env
        self.test_env = test_env

        self.algorithm = algorithm

        self.max_action = max_action
        self.min_action = min_action

        self.discrete = self.args.discrete
        self.max_step = self.args.max_step

        self.eval = self.args.eval
        self.eval_episode = self.args.eval_episode
        self.eval_step = self.args.eval_step

        self.episode = 0
        self.episode_reward = 0
        self.total_step = 0
        self.local_step = 0
        self.eval_num = 0
        self.test_num = 0

        self.T_terminal = self.env.spec.max_episode_steps

        if self.args_tester is None:
            self.render = self.args.render
            self.path = self.args.path
        else:
            self.render = self.args_tester.render
            self.path = self.args_tester.path
            self.test_episode = self.args_tester.test_episode

        # score
        self.score = 0

        self.buffer4model = Buffer(state_dim=state_dim*self.n_history, action_dim=action_dim*(self.n_history+1),
                                   max_size=args.buffer_size, on_policy=False, device=self.algorithm.device)

        """
        fail score for the environments which don't have uncertain criterion about fail
        fail score = maximum score * 0.2
        
        Half cheetah    : 1183.44
        Ant             : 647.3822
        """

        self.fail_score = 2 * 647.3822

    def evaluate(self):
        self.eval_num += 1
        episode = 0
        reward_list = []
        alive_cnt = 0

        while True:
            self.local_step = 0
            if episode >= self.eval_episode:
                break
            episode += 1
            eval_reward = 0
            observation = self.test_env.reset()
            done = False

            while not done:
                self.local_step += 1
                action = self.algorithm.eval_action(observation)
                env_action = denormalize(action, self.max_action, self.min_action)

                next_observation, reward, done, _ = self.test_env.step(env_action)

                if self.render:
                    self.test_env.render()

                eval_reward += reward
                observation = next_observation

                if self.local_step == self.T_terminal:
                    alive_cnt += 1

            reward_list.append(eval_reward)
        score_now = sum(reward_list) / len(reward_list)
        alive_rate = alive_cnt / self.eval_episode
        _score = save_policys(self.algorithm, self.score, score_now, alive_rate, self.path)

        if _score is not None:
            self.score = _score

        if self.total_step == self.args.buffer_size:
            self.buffer4model.save_buffer(self.path, 'by_full')
        if self.total_step > self.args.buffer_size and self.total_step % int(self.args.buffer_size/4) == 0:
            self.buffer4model.save_buffer(self.path, 'after_full')

        print("Eval  | Average Reward {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f}, alive rate : {:.2f}"
              .format(sum(reward_list)/len(reward_list), max(reward_list), min(reward_list), np.std(reward_list), 100*alive_rate))
        self.test_env.close()

    def run(self):
        reward_list = []
        while True:
            if self.total_step > self.max_step:
                print("Training finished")
                break

            self.episode += 1
            self.episode_reward = 0
            self.local_step = 0

            observation = self.env.reset()
            self.test_env.reset()
            done = False

            self.mem_observation = np.concatenate([observation] * self.n_history)
            self.mem_next_observation = np.concatenate([observation] * self.n_history)
            self.mem_action = np.concatenate([np.zeros(self.action_dim)] * (self.n_history+1), axis=-1)

            while not done:
                self.local_step += 1
                self.total_step += 1

                if self.render:
                    self.env.render()

                if self.total_step <= self.algorithm.training_start:
                    env_action = self.env.action_space.sample()
                    action = normalize(env_action, self.max_action, self.min_action)
                else:
                    if self.algorithm.buffer.on_policy == False:
                        action = self.algorithm.get_action(observation)
                    else:
                        action, log_prob = self.algorithm.get_action(observation)
                    env_action = denormalize(action, self.max_action, self.min_action)
                next_observation, reward, done, info = self.env.step(env_action)

                self.mem_observation[self.state_dim:] = self.mem_observation[:self.state_dim * (self.n_history - 1)]
                self.mem_observation[:self.state_dim] = observation
                self.mem_next_observation[self.state_dim:] = self.mem_observation[:self.state_dim * (self.n_history - 1)]
                self.mem_next_observation[:self.state_dim] = next_observation
                self.mem_action[self.action_dim:] = self.mem_action[:self.action_dim * self.n_history]
                self.mem_action[:self.action_dim] = action

                if self.local_step + 1 == 1000:
                    real_done = 0.
                else:
                    real_done = float(done)

                self.episode_reward += reward

                if self.algorithm.buffer.on_policy == False:
                    self.algorithm.buffer.add(observation, action, reward, next_observation, real_done)
                else:
                    self.algorithm.buffer.add(observation, action, reward, next_observation, real_done, log_prob)

                self.buffer4model.add(self.mem_observation, self.mem_action, 0., self.mem_next_observation, 0.)

                observation = next_observation

                if self.total_step >= self.algorithm.training_start and done:
                    loss_list = self.algorithm.update(self.algorithm.training_step)

                if self.eval is True and self.total_step % self.eval_step == 0:
                    self.evaluate()
                    if self.args.numpy is False:
                        df = pd.DataFrame(reward_list)
                        df.to_csv(self.path + "saved_log/reward" + ".csv")
                    else:
                        df = np.array(reward_list)
                        plot_fig(np.arange(self.total_step), reward_list, self.path + "saved_log/reward.png")
                        np.save(self.path + "saved_log/reward" + ".npy", df)

            reward_list.append(self.episode_reward)
            print("Train | Episode: {}, Reward: {:.2f}, Local_step: {}, Total_step: {}".format(self.episode, self.episode_reward, self.local_step, self.total_step))
        self.env.close()

    def test(self):
        self.test_num += 1
        episode = 0
        reward_list = []
        alive_cnt = 0

        while True:
            self.local_step = 0
            alive = False
            if episode >= self.test_episode:
                break
            episode += 1
            eval_reward = 0
            observation = self.test_env.reset()
            done = False

            while not done:
                self.local_step += 1
                action = self.algorithm.eval_action(observation)
                env_action = denormalize(action, self.max_action, self.min_action)

                if self.args_tester.add_noise is True and self.args_tester.noise_to == 'action':
                    env_action = add_noise(env_action, scale=self.args_tester.noise_scale)
                if self.args_tester.add_disturbance is True and self.args_tester.disturbance_to == 'action':
                    env_action = add_disturbance(env_action, self.local_step,
                                                 self.env.spec.max_episode_steps,
                                                 scale=self.args_tester.disturbance_scale,
                                                 frequency=self.args_tester.disturbance_frequency)

                next_observation, reward, done, _ = self.test_env.step(env_action)

                if self.args_tester.add_noise is True and self.args_tester.noise_to == 'state':
                    next_observation, _ = add_noise(next_observation, scale=self.args_tester.noise_scale)
                if self.args_tester.add_disturbance is True and self.args_tester.disturbance_to == 'state':
                    next_observation, _ = add_disturbance(next_observation, self.local_step,
                                                       self.test_env.spec.max_episode_steps,
                                                       scale=self.args_tester.disturbance_scale,
                                                       frequency=self.args_tester.disturbance_frequency)

                if self.render:
                    self.test_env.render()

                eval_reward += reward
                observation = next_observation

                if self.local_step == self.T_terminal and eval_reward > self.fail_score:
                    alive_cnt += 1
                    alive = True

            if eval_reward < 0:
                eval_reward = 0

            print("Eval of {}th episode  | Episode Reward {:.2f}, alive : {}".format(episode, eval_reward, alive))
            reward_list.append(eval_reward)

        print(
            "Eval of all episodes | Average Reward {:.2f}, Max reward: {:.2f}, Min reward: {:.2f}, Stddev reward: {:.2f}, alive rate : {:.2f}".format(
                sum(reward_list) / len(reward_list), max(reward_list), min(reward_list), np.std(reward_list),
                100 * (alive_cnt / self.test_episode)))
        self.test_env.close()
        return sum(reward_list) / len(reward_list),\
               max(reward_list), min(reward_list),\
               np.std(reward_list),\
               100 * (alive_cnt / self.test_episode)