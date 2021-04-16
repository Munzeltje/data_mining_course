# -*- coding: utf-8 -*-
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def validate(model, dataloader, config):
    model.eval()
    predictions = []
    _labels = []
    metric = nn.MSELoss()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            classification = model(inputs)
            #classification = torch.argmax(classification, dim=2)
            predictions += list(classification.view(-1).cpu().numpy())
            _labels += list(labels.view(-1).cpu().numpy())

    #print(np.round(np.array(predictions[:40]), 1))
    #print(np.around(np.array(_labels[:40]),1))
    print(metric(torch.tensor(_labels), torch.tensor(predictions)))

    predictions = np.around(np.array(predictions),1)*10 -5
    _labels = np.around(np.array(_labels),1)*10 -5
    unique_preds = np.unique(predictions)
    unique_labels = np.unique(_labels)
    #print("uniques: ", unique_labels)
    #print("unqiues pred:", unique_preds)
    print(classification_report(_labels, predictions))
