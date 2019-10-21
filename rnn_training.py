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
import reverser     # I symlinked code-1.py to reverser.py to be able to import it. Hyphens are illegal.

if __name__ == "__main__":
    # Any random seed.
    np.random.seed(None)
    savefile = 'models/lstm_test_131i256h1l'
    overwrite = False

    rnn = RNN_Reverser(use_cuda=True, hidden_dim=20, in_dim=10, out_dim=10, max_len=15, num_layers=2)
    reverser = Reverser(rnn, optimizer='Adam', lr=0.0001)

    if os.path.isfile(savefile + '.pt') and not overwrite:
        reverser.backend.load_state_dict(torch.load(savefile + '.pt'))
    if os.path.isfile(savefile + '_loss.npy') and not overwrite:
        old_loss = np.load(savefile + '_loss.npy')
    else:
        old_loss = np.array([])

    lims = (8,64)
    losses = reverser.train(batch_size=50, num_batch=200000, seq_lim=lims)

    torch.save(reverser.backend.state_dict(), savefile + '.pt')
    loss_arr = np.array(losses)
    loss_arr = np.concatenate((old_loss, loss_arr))
    np.save(savefile + '_loss.npy', loss_arr)

    for i in range(40):
        seqlen = np.random.randint(lims[0], lims[1]+1)
        xtest, ytest = reverser.generate_seqs(1, seqlen)
        print(f'input: {xtest.flatten().cpu().numpy()}')
        print(f'goal: {ytest.flatten().cpu().numpy()}')
        yhat = reverser.reverse(xtest)
        print(f'pred: {yhat.flatten().cpu().numpy()}')
        print('_________________________')
    plt.semilogy(loss_arr)
    plt.show()
