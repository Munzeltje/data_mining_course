# -*- coding: utf-8 -*-
import os, sys

import random
import torch
import numpy  as np
from torch.utils.data import Dataset
import torch.utils.data as data
import json
import pandas as pd

class MoodLoader(Dataset):
    def __init__(self, sequence_length=32, predict=False):

        self.sequence_length = sequence_length
        with open(data_path, 'r') as file:
            self.data = pd.read_csv(file)


        self.languages.sort()
        print("Number of languages: ", len(self.languages))

        self.predict = predict
        self.prediction_offset = predict_offset

    def __getitem__(self, index):
        """
        Get a random patient.
        Get a random sequence of sequence length within the data of that patient.
        On average we should have covered all the data from all the patients.

        """


        return inputs, target


    def __len__(self):
        return len(self.lines)
