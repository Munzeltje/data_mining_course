import torch
import torch.nn as nn

class LinearUnit(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(size_in,size_out), nn.ReLU())

    def forward(self,x):
        return self.network(x)

class PointwiseLTRModel(nn.Module):
    def __init__(self, features, layers):

        super().__init__()
        assert(layers < 5), "Too many layers. This should be a shallow network"
        layers = []
        start_number = {1: 8, 2: 16, 3: 32, 4: 64}[layers]
        layers.append(LinearUnit(features, start_number))
        for n in range(layers-1):
            layers.append(LinearUnit(start_number, int(start_number/2)))
            start_number = int(start_number/2)
        layers.append(nn.Linear(start_number, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
