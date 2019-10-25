"""
Author: Cedric Flamant
CS 281 Set 3
Exercise 1 and 2, RNN, LSTM, LSTM+NeuralStack to reverse strings

This script serves as a template for training Reverser with RNN
backends
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import os
import reverser     # I symlinked code-1.py to reverser.py to be able to import it. Hyphens are illegal.

if __name__ == "__main__":
    # Any random seed.
    np.random.seed(None)

    savefile = 'models/vlonglstm_256h_2l_50b_Adam_1e-3'
    overwrite = False

    # hyperparameters
    hidden_dim = 256    # Hidden size
    num_layers = 2      # number of layers to stack
    batch_size = 50     # batch size
    optimizer = 'Adam'  # Optimizer to use
    lr = 0.001          # Learning rate
    num_batch = 100000  # Number of batches between saves
    sessions = 2        # How many times to run num_batch (saving at the end of each)

    # fixed parameters (not to be modified)
    lims = (8,64)   # train sequence length range
    in_dim = 131    # 128 chars plus start, separator, and terminator
    out_dim = 131 
    max_len = 128   # maximum output length (not including terminator)

    lstm = reverser.LSTM_Reverser(use_cuda=True, hidden_dim=hidden_dim, in_dim=in_dim, out_dim=out_dim, max_len=max_len, num_layers=num_layers)
    rev = reverser.Reverser(lstm, optimizer=optimizer, lr=lr)

    ####################################################################################
    ### Training section. 
    ####################################################################################

    # if existing model parameters exist, load them. Otherwise start fresh.
    if os.path.isfile(savefile + '.pt') and not overwrite:
        print("Loading previous model")
        rev.backend.load_state_dict(torch.load(savefile + '.pt'))

    rev.coarse_scores()
