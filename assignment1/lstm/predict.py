# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.optim as optim
from LSTM import Model
from utils.utils import *

from data.load_data import get_wili_data, get_wili_data_bytes
from utils.config import LSTM_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model, validation_loader, validation_data, config, filename):
    model.eval()


    print("Iterators Done")

def main():

    config = LSTM_config()

    model = Model(config.input_dim, config.embedding_dim,
                  config.hidden_dim)

    # Load Data
    if config.input == 'bytes':
        _, validation_data = get_wili_data_bytes(config)

    else:
        _, validation_data = get_wili_data(config)
        #raise ValueError()


    validation_loader = DataLoader(validation_data,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False)

    if config.model_checkpoint is not None:
        with open(config.model_checkpoint, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)
            print("Model Loaded From: {}".format(config.model_checkpoint))

    model = model.to(device)
    predict(model, validation_loader, validation_data, config, config.val_data_path.replace('bytes_', ''))


if __name__ == '__main__':
    main()
