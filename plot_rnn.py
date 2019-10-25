"""
Author: Cedric Flamant
CS 281 Set 3
Exercise 1 and 2, RNN, LSTM, LSTM+NeuralStack to reverse strings

This is a script for plotting RNN run results.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

    loss_freq = 100
    acc_freq = 10000

    filenames = ['models/rnn_256h_1l_50b_Adam_1e-3',
                 'models/rnn_256h_1l_50b_Adam_1e-4',
                 'models/rnn_256h_1l_50b_RMSprop_1e-3',
                 'models/rnn_256h_1l_50b_RMSprop_1e-4',
                 'models/rnn_256h_1l_50b_SGD_1e-3',
                 'models/rnn_256h_1l_50b_SGD_1e-4'
                ]

    labels = ["Adam, lr=0.001",
              "Adam, lr=0.0001",
              "RMSprop, lr=0.001",
              "RMSprop, lr=0.0001",
              "SGD, lr=0.001",
              "SGD, lr=0.0001"]
    losses, val_accs, test_accs = [], [], []
    lossbatch, accbatch = [], []
    for i,fname in enumerate(filenames):
        losses.append(np.load(fname + '_loss.npy'))
        val_accs.append(np.load(fname + '_valaccs.npy'))
        test_accs.append(np.load(fname + '_testaccs.npy'))
        lossbatch.append(np.arange(losses[0].shape[0])*loss_freq)
        accbatch.append(np.arange(val_accs[0].shape[0])*acc_freq)

    #figloss, axloss = plt.subplots(1,1)
    #for i, fname in enumerate(filenames):
    #    axloss.plot(lossbatch[i], losses[i], label=labels[i])
    #axloss.legend()
    #axloss.set_title("Losses")

    figval, axval = plt.subplots(1,1)
    for i, fname in enumerate(filenames):
        axval.plot(accbatch[i], val_accs[i], label=labels[i])
    axval.legend()
    axval.set_title("RNN 256h 1l 50b, Validation Accuracy")
    axval.set_ylabel("fine accuracy")
    axval.set_xlabel("batch number")
    figval.savefig('plots/rnnbig_val.pdf', bbox_inches='tight')

    figtest, axtest = plt.subplots(1,1)
    for i, fname in enumerate(filenames):
        axtest.plot(accbatch[i], test_accs[i], label=labels[i])
    axtest.legend()
    axtest.set_title("RNN 256h 1l 50b, Test Accuracy")
    axtest.set_ylabel("fine accuracy")
    axtest.set_xlabel("batch number")
    figtest.savefig('plots/rnnbig_test.pdf', bbox_inches='tight')

    plt.show()

