# -*- coding: utf-8 -*-
import os, sys

import random
import torch
import numpy  as np
from torch.utils.data import Dataset
import torch.utils.data as data
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import dump
import joblib


class MoodLoader(Dataset):
    def __init__(self, data_path, config, sequence_length=2, predict=False):

        self.sequence_length = sequence_length
        self.config = config

        #self.train = pd.read_csv("../data/dataset_train.csv")
        #self.val = pd.read_csv("../data/dataset_train.csv")
        #self.test = pd.read_csv("../data/dataset_train.csv")
        self.data = pd.read_csv(data_path)
        self.numpy_data = self.data.drop(['id', 'date'], axis=1)
        self.numpy_data = np.array(self.numpy_data)[:,:config.input_dim]

        self.users = self.data['id'].unique()
        self.len = len(self.data)
        self.predict = predict
        self.scaler = StandardScaler().fit(self.numpy_data)
        dump(self.scaler, "scaler_train.joblib")
        if self.predict:
            self.len = self.len*4
            self.scaler = joblib.load("scaler_train.joblib")


    def __getitem__(self, index):
        """
        Get a random patient.
        Get a random sequence of sequence length within the data of that patient.
        On average we should have covered all the data from all the patients.

        """
        rand_idx = np.random.randint(0,len(self.users),1)
        user = self.users[rand_idx][0]
        df_user = self.data[self.data['id'] == user]
        df_user = df_user.drop(['id', 'date'], axis=1)
        df_user = np.array(df_user)
        rand_seq_start = np.random.randint(0,df_user.shape[0]-self.sequence_length-1,1)[0]
        inputs = df_user[rand_seq_start:rand_seq_start+self.sequence_length,:self.config.input_dim]
        inputs = self.scaler.transform(inputs)
        #targets = np.around(df_user[rand_seq_start+1:rand_seq_start+self.sequence_length+1, 0])
        targets = df_user[rand_seq_start+1:rand_seq_start+self.sequence_length+1, 0]/10
        return inputs, targets


    def __len__(self):
        return self.len
