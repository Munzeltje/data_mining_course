# -*- coding: utf-8 -*-
from .mood_data import *

def get_mood_data(config):
    """
    get train and validation data loaders.
    """

    data_path = config.data_path
    label_path = config.label_path

    val_data_path = config.val_data_path
    val_label_path = config.val_label_path

    train_data = MoodLoader(data_path=data_path,
                          label_path=label_path,
                          sequence_length=config.sequence_length
                         )

    val_data = MoodLoader(data_path=val_data_path,
                          label_path=val_label_path,
                          sequence_length=config.sequence_length
                         )

    return train_data, val_data
