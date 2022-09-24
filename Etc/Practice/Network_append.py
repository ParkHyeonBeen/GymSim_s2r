import torch
import torch.nn as nn
import torchbnn as bnn
import torch.nn.functional as F
import torch.optim as optim
from Common.Utils import *

class net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=(256, 256)):
        super(net, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.network_reminder = nn.ModuleList([nn.Linear(state_dim, hidden_dim[0]), nn.ReLU()])
        for i in range(2):
            self.network_reminder.append(nn.Linear(hidden_dim[0], hidden_dim[0]))
            self.network_reminder.append(nn.ReLU())
        self.network_last_layer = nn.Linear(hidden_dim[-1], action_dim)
        self.network = self.network_reminder.append(self.network_last_layer)

        self.dynamics_net_inner = nn.ModuleList([
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=(self.state_dim + self.action_dim), out_features=256),
            nn.ReLU()])

        self.target = 0.0
        self.target = torch.tensor(self.target, dtype=torch.float)

        self.optimizer_reminder = optim.Adam(self.network_reminder.parameters(), lr=0.001)
        self.optimizer_last_layer = optim.Adam(self.network_last_layer.parameters(), lr=0.001)

    def forward(self, inp):
        z = torch.tensor(inp, dtype=torch.float)

        for i in range(len(self.network)):
            z = self.network[i](z)
        return z

    def train_net(self, inp):

        loss = abs(inp - self.target)

        with torch.autograd.set_detect_anomaly(True):
            self.optimizer_reminder.zero_grad()
            self.optimizer_last_layer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer_reminder.step()
            for _ in range(10):
                self.optimizer_last_layer.step()

if __name__ == '__main__':
    a = net(3, 1)

    data = []
    for i in range(3000):
        inp = a([5, 6, 7])
        a.train_net(inp)
        inp = inp.detach().numpy()
        plot_data(inp)
        # plt.plot(data)
        # plt.show(block=False)
        # plt.pause(0.0001)
        # plt.cla()

