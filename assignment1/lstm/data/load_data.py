# -*- coding: utf-8 -*-
from .mood_data import *

def get_mood_data(config):
    """
    get train and validation data loaders.
    """

    data_path = config.data_path

    val_data_path = config.val_data_path

    train_data = MoodLoader(data_path=data_path, config=config,
                          sequence_length=config.sequence_length
                         )

    val_data = MoodLoader(data_path=val_data_path, config=config,
                          sequence_length=config.sequence_length, predict=True
                         )

    return train_data, val_data
