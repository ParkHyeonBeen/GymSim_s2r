import numpy as np
import torch
import torch.nn as nn
import torchbnn as bnn
import torch.nn.functional as F
from Common.Utils import weight_init
import collections

class Ensemble:
    def __init__(self, base_model, ensemble_size = 5, buffer_size = 256):
        self.model_ensemble = base_model
        self.ensemble_size = ensemble_size

        self.buffer = collections.deque(maxlen=buffer_size)
        self.ensemble_list = []

        for layer in self.model_ensemble:
            if not isinstance(layer, nn.ReLU):
                if isinstance(layer, bnn.BayesLinear):
                    layer.weight_mu.data.fill_(0.0)
                    layer.weight_log_sigma.data.fill_(0.0)

                    layer.bias_mu.data.fill_(0.0)
                    layer.bias_log_sigma.data.fill_(0.0)
                else:
                    layer.weight.data.fill_(0.0)
                    layer.bias.data.fill_(0.0)

    def add(self, model, cost):
        self.buffer.append((model, cost))

    def get_best(self):
        self.ensemble_list = []
        self._select_bests()
        for selected in self.ensemble_list:
            model, cost = selected
            self._put_to_model(model)

        return self.model_ensemble

    def _select_bests(self):
        for i, data in enumerate(self.buffer):
            if i < self.ensemble_size:
                self.ensemble_list.append(data)
            elif i == self.ensemble_size - 1:
                self.ensemble_list.sort(key=lambda x: -x[1])
            else:
                if self.ensemble_list[0][1] > data[1]:
                    self.ensemble_list[0] = data
                    self.ensemble_list.sort(key=lambda x: -x[1])
                else:
                    pass

    def _put_to_model(self, model):

        for i, layer in enumerate(model):
            if not isinstance(layer, nn.ReLU):
                if isinstance(layer, bnn.BayesLinear):
                    self.model_ensemble[i].weight_mu = nn.Parameter(
                        self.model_ensemble[i].weight_mu + layer.weight_mu.data / self.ensemble_size)
                    self.model_ensemble[i].weight_log_sigma = nn.Parameter(
                        self.model_ensemble[i].weight_log_sigma + layer.weight_log_sigma.data / self.ensemble_size)

                    self.model_ensemble[i].bias_mu = nn.Parameter(
                        self.model_ensemble[i].bias_mu + layer.bias_mu.data / self.ensemble_size)
                    self.model_ensemble[i].bias_log_sigma = nn.Parameter(
                        self.model_ensemble[i].bias_log_sigma + layer.bias_log_sigma.data / self.ensemble_size)
                else:
                    self.model_ensemble[i].weight = nn.Parameter(
                        self.model_ensemble[i].weight + layer.weight.data / self.ensemble_size)
                    self.model_ensemble[i].bias = nn.Parameter(
                        self.model_ensemble[i].bias + layer.bias.data / self.ensemble_size)

    def put_to_target(self, target):

        for i, layer in enumerate(self.model_ensemble):
            if not isinstance(layer, nn.ReLU):
                if isinstance(layer, bnn.BayesLinear):
                    target[i].weight_mu = nn.Parameter(layer.weight_mu.data)
                    target[i].weight_log_sigma = nn.Parameter(layer.weight_log_sigma.data)

                    target[i].bias_mu = nn.Parameter(layer.bias_mu.data)
                    target[i].bias_log_sigma = nn.Parameter(layer.bias_log_sigma.data)
                else:
                    target[i].weight = nn.Parameter(layer.weight.data)
                    target[i].bias = nn.Parameter(layer.bias.data)



