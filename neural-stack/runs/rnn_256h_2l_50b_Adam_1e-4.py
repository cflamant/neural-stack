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
from .. import reverser

if __name__ == "__main__":
    # Any random seed.
    np.random.seed(None)

    savefile = 'neural-stack/models/rnn_256h_2l_50b_Adam_1e-4'
    overwrite = False

    # hyperparameters
    hidden_dim = 256    # Hidden size
    num_layers = 2      # number of layers to stack
    batch_size = 50     # batch size
    optimizer = 'Adam'  # Optimizer to use
    lr = 0.0001          # Learning rate
    num_batch = 100000  # Number of batches between saves
    sessions = 2        # How many times to run num_batch (saving at the end of each)

    # fixed parameters (not to be modified)
    lims = (8,64)   # train sequence length range
    in_dim = 131    # 128 chars plus start, separator, and terminator
    out_dim = 131 
    max_len = 128   # maximum output length (not including terminator)

    rnn = reverser.RNN_Reverser(use_cuda=True, hidden_dim=hidden_dim, in_dim=in_dim, out_dim=out_dim, max_len=max_len, num_layers=num_layers)
    rev = reverser.Reverser(rnn, optimizer=optimizer, lr=lr)

    ####################################################################################
    ### Training section. 
    ####################################################################################

    # if existing model parameters exist, load them. Otherwise start fresh.
    if os.path.isfile(savefile + '.pt') and not overwrite:
        print("Loading previous model")
        rev.backend.load_state_dict(torch.load(savefile + '.pt'))
    if os.path.isfile(savefile + '_loss.npy') and not overwrite:
        old_loss = np.load(savefile + '_loss.npy')
    else:
        old_loss = np.array([])
    if os.path.isfile(savefile + '_valaccs.npy') and not overwrite:
        old_valaccs = np.load(savefile + '_valaccs.npy')
    else:
        old_valaccs = np.array([])
    if os.path.isfile(savefile + '_testaccs.npy') and not overwrite:
        old_testaccs = np.load(savefile + '_testaccs.npy')
    else:
        old_testaccs = np.array([])

    for i in range(sessions):
        islast = (i == sessions-1)
        losses, val_accs, test_accs = rev.train(batch_size=batch_size, num_batch=num_batch, seq_lim=lims, last=islast)

        torch.save(rev.backend.state_dict(), savefile + '.pt')
        loss_arr = np.array(losses)
        loss_arr = np.concatenate((old_loss, loss_arr))
        np.save(savefile + '_loss.npy', loss_arr)
        valaccs_arr = np.array(val_accs)
        valaccs_arr = np.concatenate((old_valaccs, valaccs_arr))
        np.save(savefile + '_valaccs.npy', valaccs_arr)
        testaccs_arr = np.array(test_accs)
        testaccs_arr = np.concatenate((old_testaccs, testaccs_arr))
        np.save(savefile + '_testaccs.npy', testaccs_arr)

        old_loss = loss_arr
        old_valaccs = valaccs_arr
        old_testaccs = testaccs_arr
        print(f"session {i}, Checkpoint saved")

    # A sampling of reversing performance at the end
    for i in range(20):
        seqlen = np.random.randint(lims[0], lims[1]+1)
        xtest, ytest = rev.generate_seqs(1, seqlen)
        print(f'input: {xtest.flatten().cpu().numpy()}')
        print(f'goal: {ytest.flatten().cpu().numpy()}')
        yhat = rev.reverse(xtest)
        print(f'pred: {yhat.flatten().cpu().numpy()}')
        print('_________________________')

    #fig, ax = plt.subplots(3,1)
    #ax[0].plot(loss_arr)
    #ax[1].plot(valaccs_arr)
    #ax[2].plot(testaccs_arr)
    #plt.show()
