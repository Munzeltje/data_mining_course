# -*- coding: utf-8 -*-
import numpy as np
import sys

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.optim as optim

from LSTM import Model
from data.load_data import get_mood_data
from utils.config import LSTM_config
from utils.utils import validate

from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, training_loader, validation_loader, val_data, config, model_name="Crypto_ETH"):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)
    #CE = nn.CrossEntropyLoss(weight=torch.tensor([10,10,230,57,29,2.3,1.0,2.86,92.2,10]).float().to(device))
    CE = nn.MSELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
    avg_train_loss = []
    train_loss = []
    for epoch in range(config.epochs):
        for i, (inputs, labels) in enumerate(training_loader):
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            optimizer.zero_grad()
            classification = model(inputs).squeeze(2)
            labels_classification = labels.float()
            class_loss = CE(classification, labels_classification)

            loss = class_loss
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        if  (epoch+1) % config.print_every == 0:
            print("Current Epoch: {}".format(epoch+1))

            avg = np.mean(train_loss[-100:])
            avg_train_loss.append(avg)
            print('Loss: %.6f' % avg)
            print()
            validate(model,validation_loader, config)
            model.train()

        scheduler.step()

        #torch.save(model.state_dict(), "./models/LSTM/"+str(epoch)+"_"+str(config.batch_size)+"_"+str(config.input)+"_"+str(config.sequence_length)+".pt")

        model.train()

    print("Iterators Done")

def main():

    config = LSTM_config()

    training_data, validation_data = get_mood_data(config)

    training_loader = DataLoader(training_data,
                             batch_size=config.batch_size,
                             shuffle=True,
                             drop_last=False)

    val_loader = DataLoader(validation_data,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False)

    model = Model(config.input_dim, config.hidden_dim)

    if config.model_checkpoint is not None:
        with open(config.model_checkpoint, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)
            print("Model Loaded From: {}".format(config.model_checkpoint))
    model = model.to(device)
    train(model, training_loader,val_loader, validation_data, config, model_name=config.model_name)


if __name__ == '__main__':
    main()
