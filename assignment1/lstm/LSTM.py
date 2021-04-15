# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
#from Embedding import Embedder

class Embedder(nn.Module):
    def __init__(self):
        super().__init__()

        self._day_embedding = nn.Embedding(7, 4)

    def forward(self, inputs):
        current_day = inputs[:,:,0].long()

        day_emb = self._day_embedding(current_day)

        return day_emb


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, bidirectional=False, sequence_length=32, validate=False):
        super().__init__()
        self._bidirectional = 2 if bidirectional else 1
        self._num_layers = num_layers
        self._hidden_dim = hidden_dim
        self._validate = validate
        self._linear_dim = self._hidden_dim*self._bidirectional
        hidden_layer_dim = self._linear_dim // 2
        #self._embedding = Embedder()

        self._lstm = nn.LSTM(input_dim,
                             hidden_dim,
                             num_layers,
                             dropout=0.3,
                             bidirectional=bidirectional,
                             batch_first=True)

        self.output_sequence = nn.Sequential(nn.Linear(self._linear_dim, hidden_layer_dim),
                                           nn.LeakyReLU(0.2), nn.Dropout(0.5))

        #self.price_layer = nn.Linear(hidden_layer_dim, 1)
        self.classification = nn.Linear(hidden_layer_dim, 3)

    def forward(self, inputs, h=None, c=None, validate=False):
        self._validate = validate
        #emb = self._embedding(inputs)

        if self._validate:
            out, (h, c) = self._lstm(inputs, (h, c))
        else:
            out, _ = self._lstm(inputs)

        output = self.output_sequence(out)

        #logits_prices = self.price_layer(output)
        logits_classification = self.classification(output)

        return logits_classification, (h, c)
