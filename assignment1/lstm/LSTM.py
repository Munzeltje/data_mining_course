# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._lstm = nn.LSTM(input_dim,
                             hidden_dim,
                             batch_first=True)

        self.classification = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())


    def forward(self, inputs, h=None, c=None, validate=False):
        out, _ = self._lstm(inputs)
        logits_classification = self.classification(out)

        return logits_classification
