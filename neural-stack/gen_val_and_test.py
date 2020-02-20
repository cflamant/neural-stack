"""
Author: Cedric Flamant
CS 281 Set 3
Exercise 1 and 2, RNN, LSTM, LSTM+NeuralStack to reverse strings

Generates validation and test sets. Intended to be run once!
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import sys
import os
from . import reverser
import csv


if __name__ == "__main__":
    # Any random seed.
    np.random.seed(None)
    savefile = 'neural-stack/data/'

    # Whether to overwrite validation and test sets if already present
    overwrite = False

    # Number of examples to use in both validation and test sets
    num_samples = 1000

    # Validation sequence lengths (inclusive)
    val_seq_lim = (8, 64)

    # Test sequence lengths (inclusive)
    test_seq_lim = (65, 128)

    # Create a dummy backend. We just make this to access the generate_seqs() method.
    rnn = reverser.RNN_Reverser()
    rev = reverser.Reverser(rnn)

    if os.path.isfile(savefile + 'val_x.txt') or os.path.isfile(savefile + 'val_y.txt'):
        if not overwrite:
            print("Validation data already exists. Either delete it or set overwrite")
            print("flag to True.")
            sys.exit()

    val_x, val_y = [], []
    for i in range(num_samples):
        seq_len = np.random.randint(val_seq_lim[0],val_seq_lim[1]+1)
        x_tensor, y_tensor = rev.generate_seqs(1, seq_len)
        x_tensor = torch.flatten(x_tensor)
        y_tensor = torch.flatten(y_tensor)
        x = x_tensor.numpy()
        y = y_tensor.numpy()
        val_x.append(x)
        val_y.append(y)
    with open(savefile + 'val_x.txt', 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        for x in val_x:
            writer.writerow(x)
    with open(savefile + 'val_y.txt', 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        for y in val_y:
            writer.writerow(y)

    if os.path.isfile(savefile + 'test_x.txt') or os.path.isfile(savefile + 'test_y.txt'):
        if not overwrite:
            print("Test data already exists. Either delete it or set overwrite")
            print("flag to True.")
            sys.exit()

    test_x, test_y = [], []
    for i in range(num_samples):
        seq_len = np.random.randint(test_seq_lim[0],test_seq_lim[1]+1)
        x_tensor, y_tensor = rev.generate_seqs(1, seq_len)
        x_tensor = torch.flatten(x_tensor)
        y_tensor = torch.flatten(y_tensor)
        x = x_tensor.numpy()
        y = y_tensor.numpy()
        test_x.append(x)
        test_y.append(y)
    with open(savefile + 'test_x.txt', 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        for x in test_x:
            writer.writerow(x)
    with open(savefile + 'test_y.txt', 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        for y in test_y:
            writer.writerow(y)
