"""
Author: Cedric Flamant
CS 281 Set 3
Exercise 1 and 2, RNN, LSTM, LSTM+NeuralStack to reverse strings

This script loads the validation and test data.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import os



def get_data(device):
    """Read saved validation and test data

    Parameters
    ----------
    device : torch.device()
        The device the model is currently using, i.e. cuda or cpu

    Returns
    -------
    val_x : List[LongTensor of dim (*, 1)]
        Validation set int-encoded input sequences
    val_y : List[ndarray of dim (*)]
        Validation set int-encoded output sequences
    test_x : List[LongTensor of dim (*, 1)]
        Test set int-encoded input sequences
    test_y : List[ndarray of dim (*)]
        Test set int-encoded output sequences
    """
    savefile = 'data/'

    val_x, val_y = [], []
    test_x, test_y = [], []

    if os.path.isfile(savefile + 'val_x.txt') and os.path.isfile(savefile + 'val_y.txt'):
        with open(savefile + 'val_x.txt', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                x_np = np.asarray(row,dtype=np.int64)
                x_tensor = torch.unsqueeze(torch.as_tensor(x_np, device=device), 1)
                val_x.append(x_tensor)
        with open(savefile + 'val_y.txt', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                val_y.append(np.asarray(row,dtype=np.int64))

    if os.path.isfile(savefile + 'test_x.txt') and os.path.isfile(savefile + 'test_y.txt'):
        with open(savefile + 'test_x.txt', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                x_np = np.asarray(row,dtype=np.int64)
                x_tensor = torch.unsqueeze(torch.as_tensor(x_np, device=device), 1)
                test_x.append(x_tensor)
        with open(savefile + 'test_y.txt', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                test_y.append(np.asarray(row,dtype=np.int64))
    return val_x, val_y, test_x, test_y

