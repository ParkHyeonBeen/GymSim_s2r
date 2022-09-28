import torch
import torch.nn.functional as F
import numpy as np

from Common.Buffer import Buffer
from Common.Utils import copy_weight, soft_update
from Network.Basic_Network import Q_Network
from Network.Gaussian_Actor import Squashed_Gaussian_Actor
from Network.Model_Network import *
import pdb

class SAC_v3:
    def __init__(self, state_dim, action_dim, replay_buffer, args, device):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_history = args.n_history

        self.buffer = replay_buffer
        self.args = args
        self.device = device

        self.worker_step = torch.zeros(1, dtype=torch.int32, requires_grad=False, device='cpu')
        self.update_step = torch.zeros(1, dtype=torch.int32, requires_grad=False, device='cpu')
        self.eps = torch.zeros(1, dtype=torch.int32, requires_grad=False, device='cpu')

        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma
        self.training_start = args.training_start
        self.current_step = 0
        self.critic_update = args.critic_update

        self.target_entropy = -action_dim
        self.log_alpha = torch.as_tensor(np.log(args.alpha), dtype=torch.float32, device=self.device).requires_grad_()
        self.optimize_alpha = args.train_alpha

        self.actor = Squashed_Gaussian_Actor(self.state_dim, self.action_dim, args.hidden_dim, args.log_std_min, args.log_std_max).to(self.device)
        self.critic1 = Q_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.critic2 = Q_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target_critic1 = Q_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.target_critic2 = Q_Network(self.state_dim, self.action_dim, args.hidden_dim).to(self.device)
        self.imn = InverseModelNetwork(self.state_dim, self.action_dim, args, hidden_dim=args.model_hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=args.critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=args.critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)
        self.imn_optimizer = torch.optim.Adam(self.imn.parameters(), lr=args.model_lr)
        copy_weight(self.critic1, self.target_critic1)
        copy_weight(self.critic2, self.target_critic2)

        self.name = 'SAC_v3'

        self.lambda_t = args.lambda_t
        self.lambda_s = args.lambda_s
        self.eps_p = args.eps_p

        self.train()
        self.imn.trains()
        self.imn_criterion = nn_ard.ELBOLoss(self.imn, F.smooth_l1_loss).to(device)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic1.train(training)
        self.critic2.train(training)
        self.target_critic1.train(training)
        self.target_critic2.train(training)

    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        self.target_critic1.eval()
        self.target_critic2.eval()

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def get_action(self, state):
        with torch.no_grad():
            state = np.expand_dims(np.array(state), axis=0)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action, _ = self.actor(state)

        return action.cpu().numpy()[0]

    def eval_action(self, state):
        with torch.no_grad():
            state = np.expand_dims(np.array(state), axis=0)
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            action, _ = self.actor(state, deterministic=True)

        return action.cpu().numpy()[0]

    def train_alpha(self, s):
        _, s_logpi = self.actor(s)
        alpha_loss = -(self.log_alpha.exp() * (s_logpi + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

    def train_critic(self, s, a, r, ns, d):
        with torch.no_grad():
            ns_action, ns_logpi = self.actor(ns)
            target_min_aq = torch.min(
                self.target_critic1(ns, ns_action),
                self.target_critic2(ns, ns_action))
            target_q = (r + self.gamma * (1 - d) * (target_min_aq - self.alpha * ns_logpi)).detach()

        critic1_loss = F.mse_loss(input=self.critic1(s, a), target=target_q)
        critic2_loss = F.mse_loss(input=self.critic2(s, a), target=target_q)

        self.critic1_optimizer.zero_grad()
        try:
            critic1_loss.backward()
        except Exception:
            pdb.set_trace()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        try:
            critic2_loss.backward()
        except Exception:
            pdb.set_trace()
        self.critic2_optimizer.step()

        return ns_action

    def train_actor(self, s, ns_action):
        random_s = torch.normal(s, self.eps_p)
        s_action, s_logpi = self.actor(s)
        random_s_action, _ = self.actor(random_s)

        min_aq_rep = torch.min(self.critic1(s, s_action), self.critic2(s, s_action))
        policy_loss = (self.alpha * s_logpi - min_aq_rep).mean()

        policy_loss_s = torch.norm(s_action - random_s_action, dim=--1).mean()
        policy_loss_t = torch.norm(s_action - ns_action, dim=-1).mean()
        policy_loss += self.lambda_t * policy_loss_t
        policy_loss += self.lambda_s * policy_loss_s

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

    def train_imn(self, s_history, a_history, ns):
        a = a_history[:, :self.action_dim]
        prev_a = a_history[:, self.action_dim:]
        a_hat = self.imn(s_history, prev_a, ns)

        def get_kl_weight(epoch):
            if self.args.use_prev_policy is True:
                return min(1, self.args.reg_weight * epoch)
            else:
                return min(1, self.args.reg_weight * (epoch - self.args.model_train_start_step))

        if self.args.net_type == "bnn" and self.worker_step > self.args.model_train_start_step:
            model_loss = self.imn_criterion(a_hat, a, 1, get_kl_weight(self.worker_step).to(self.device)).mean()
        else:
            model_loss = F.smooth_l1_loss(a, a_hat).mean()

        self.imn_optimizer.zero_grad()
        model_loss.backward()
        self.imn_optimizer.step()

    def update(self, worker_step):

        self.current_step += 1
        if worker_step < self.args.model_train_start_step and self.args.use_prev_policy is False:
            s_history, a_history, r, ns, d = self.buffer.sample(self.batch_size)

            s = s_history[:, :self.state_dim]
            a = a_history[:, :self.action_dim]

            ns_action = self.train_critic(s, a, r, ns, d)
            self.train_actor(s, ns_action)

            if self.optimize_alpha is True:
                self.train_alpha(s)

            if self.current_step % self.critic_update == 0:
                soft_update(self.critic1, self.target_critic1, self.tau)
                soft_update(self.critic2, self.target_critic2, self.tau)
        else:
            self.imn.trains()
            s_history, a_history, _, ns, _ = self.buffer.sample(64)
            self.train_imn(s_history, a_history, ns)

