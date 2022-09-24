from Common.Ensemble_model import Ensemble
from Network.Model_Network import *


class model(nn.Module):
    def __init__(self, state_dim=1, action_dim=1, hidden_dim=(10, 10), init = False):
        super(model, self).__init__()
        self.model1 = nn.ModuleList([nn.Linear(2 * state_dim, hidden_dim[0]), nn.ReLU()])
        for i in range(len(hidden_dim) - 1):
            self.model1.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            self.model1.append(nn.ReLU())
        self.model2 = nn.Linear(hidden_dim[-1], action_dim)
        self.model3 = self.model1.append(self.model2)

        if init is True:
            for layer in self.model3:
                if not isinstance(layer, nn.ReLU):
                    layer.weight.data.fill_(0.0)
                    layer.bias.data.fill_(0.0)

        self.apply(weight_init)


# esb = Ensemble(model)

# for layer in esb.model_ensemble:
#     if not isinstance(layer, nn.ReLU):
#         print(layer.weight.data)
#         print(layer.bias.data)

# models = []
#
# for i in range(5):
#     _model = model().model3
#     # print(_model[0].weight.data)
#     models.append(_model)
#     # esb.add(_model, random.random())

def A(a, b, *models):
    for model in models:
        print(model)
    print(a, b)

A(1, 2, 1, 553, 4, 5)

# model_ensemble = esb.get_best()

# for i, layer in enumerate(model_ensemble):
#     if not isinstance(layer, nn.ReLU):
#         print(layer.weight.data)