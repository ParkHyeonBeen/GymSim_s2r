import numpy as np
import torch

class Buffer:
    def __init__(self, state_dim, action_dim, next_state_dim, args, max_size=1e6, on_policy=False, device=None):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.next_state_dim = next_state_dim
        self.on_policy = on_policy
        self.n_history = args.n_history

        self.device = device
        if self.device is None:
            assert ValueError

        if type(self.state_dim) == int:
            self.s = np.empty((self.max_size, self.state_dim), dtype=np.float32)
            self.ns = np.empty((self.max_size, self.next_state_dim), dtype=np.float32)
        else:
            self.s = np.empty((self.max_size, *self.state_dim), dtype=np.uint8)
            self.ns = np.empty((self.max_size, *self.next_state_dim), dtype=np.uint8)

        self.a = np.empty((self.max_size, self.action_dim), dtype=np.float32)
        self.r = np.empty((self.max_size, 1), dtype=np.float32)
        self.d = np.empty((self.max_size, 1), dtype=np.float32)

        if self.on_policy == True:
            self.log_prob = np.empty((self.max_size, self.action_dim), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        if self.full == False:
            return self.idx
        else:
            return self.max_size

    def add(self, s, a, r, ns, d, log_prob=None):
        np.copyto(self.s[self.idx], s)
        np.copyto(self.a[self.idx], a)
        np.copyto(self.r[self.idx], r)
        np.copyto(self.ns[self.idx], ns)
        np.copyto(self.d[self.idx], d)

        if self.on_policy == True:
            np.copyto(self.log_prob[self.idx], log_prob)

        self.idx = (self.idx + 1) % self.max_size
        if self.idx == 0:
            self.full = True

    def delete(self):
        if type(self.state_dim) == int:
            self.s = np.empty((self.max_size, self.state_dim), dtype=np.float32)
            self.ns = np.empty((self.max_size, self.next_state_dim), dtype=np.float32)
        else:
            self.s = np.empty((self.max_size, *self.state_dim), dtype=np.uint8)
            self.ns = np.empty((self.max_size, *self.next_state_dim), dtype=np.uint8)

        self.a = np.empty((self.max_size, self.action_dim), dtype=np.float32)
        self.r = np.empty((self.max_size, 1), dtype=np.float32)
        self.d = np.empty((self.max_size, 1), dtype=np.float32)

        if self.on_policy == True:
            self.log_prob = np.empty((self.max_size, self.action_dim), dtype=np.float32)

        self.idx = 0
        self.full = False

    def all_sample(self):
        ids = np.arange(self.max_size if self.full else self.idx)
        states = torch.as_tensor(self.s[ids], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.a[ids], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(self.r[ids], dtype=torch.float32, device=self.device)
        states_next = torch.as_tensor(self.ns[ids], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.d[ids], dtype=torch.float32, device=self.device)

        if self.on_policy == True:
            log_probs = torch.as_tensor(self.log_prob[ids], dtype=torch.float32, device=self.device)
            return states, actions, rewards, states_next, dones, log_probs

        return states, actions, rewards, states_next, dones

    def sample(self, batch_size):
        ids = np.random.randint(0, self.max_size if self.full else self.idx, size=batch_size)
        states = torch.as_tensor(self.s[ids], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(self.a[ids], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(self.r[ids], dtype=torch.float32, device=self.device)
        states_next = torch.as_tensor(self.ns[ids], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(self.d[ids], dtype=torch.float32, device=self.device)

        if self.on_policy == True:
            log_probs = torch.as_tensor(self.log_prob[ids], dtype=torch.float32, device=self.device)
            return states, actions, rewards, states_next, dones, log_probs

        return states, actions, rewards, states_next, dones

    def save_buffer(self, path, name):
        path = path + 'saved_buffer/buffer_' + name

        np.save(path + '_s.npy', self.s)
        np.save(path + '_a.npy', self.a)
        np.save(path + '_ns.npy', self.ns)

    def load_buffer(self, path, name):
        path = path + 'saved_buffer/buffer_' + name

        self.s = np.load(path + '_s.npy')
        self.a = np.load(path + '_a.npy')
        self.ns = np.load(path + '_ns.npy')

        self.idx = np.count_nonzero(self.s, axis=0)[0]









