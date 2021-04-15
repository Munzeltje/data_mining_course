# -*- coding: utf-8 -*-
import numpy as np
import sys

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.optim as optim

from LSTM import Model
from data.load_data import get_data
from utils.config import LSTM_config
from utils.utils import validate, validate_set

from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, training_loader, validation_loader, val_data, config, model_name="Crypto_ETH"):
    model.train()
    #optimizer = optim.Adam(model.parameters(), lr=config.lr)
    #optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.99,0.999), weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=5e-4)
    L1 = nn.MSELoss()
    CE = nn.CrossEntropyLoss(weight=torch.tensor([3,3,1]).float().to(device))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
    avg_train_loss = []
    train_loss = []
    for epoch in range(config.epochs):
        for i, (inputs, labels) in enumerate(training_loader):
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            optimizer.zero_grad()
            classification, _ = model(inputs)
            labels_classification = labels[:,:,-2].long()

            #price_loss = torch.sqrt(L1(prices.squeeze(2), labels_prices))
            class_loss = CE(classification.squeeze(2).permute(0,2,1), labels_classification)
            #class_loss = class_loss #+ 0.5*CE(classification.squeeze(2).permute(0,2,1),penalty_loss(classification))

            #loss = 100*price_loss + class_loss
            loss = class_loss
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        if  (epoch+1) % config.print_every == 0:
            #print("price loss: ",price_loss.item())
            print("Current Epoch: {}".format(epoch+1))

            avg = np.mean(train_loss[-100:])
            avg_train_loss.append(avg)
            print('Loss: %.6f' % avg)
            print()
            _ = validate_set(model, val_data, validation_loader, config)

        scheduler.step()

        torch.save(model.state_dict(), "./models/LSTM/"+str(epoch)+"_"+str(config.batch_size)+"_"+str(config.input)+"_"+str(config.sequence_length)+".pt")

        model.train()

    print("Iterators Done")

def main():

    config = LSTM_config()

    training_data, validation_data = get_data(config)

    training_loader = DataLoader(training_data,
                             batch_size=config.batch_size,
                             shuffle=True,
                             drop_last=False)

    val_loader = DataLoader(validation_data,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False)

    model = Model(config.input_dim, config.hidden_dim, config.num_layers, bidirectional=False)

    if config.model_checkpoint is not None:
        with open(config.model_checkpoint, 'rb') as f:
            state_dict = torch.load(f)
            model.load_state_dict(state_dict)
            print("Model Loaded From: {}".format(config.model_checkpoint))
    model = model.to(device)
    train(model, training_loader,val_loader, validation_data, config, model_name=config.model_name)


if __name__ == '__main__':
    main()
