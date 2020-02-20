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
    acc_freq = 1000

    filenames = ['neural-stack/models/stack_16h_16e_1l_50b_Adam_1e-3_accfreq1000',
                 'neural-stack/models/stack_64h_1e_1l_50b_Adam_1e-3',
                 'neural-stack/models/manystack_64h_64e_1l_50b_Adam_1e-3_5'
                ]

    labels = ["16h, 16e",
              "64h, 1e",
              "64h, 64e"
              ]
    losses, val_accs, test_accs = [], [], []
    lossbatch, accbatch = [], []
    for i,fname in enumerate(filenames):
        losses.append(np.load(fname + '_loss.npy'))
        val_accs.append(np.load(fname + '_valaccs.npy'))
        test_accs.append(np.load(fname + '_testaccs.npy'))
        lossbatch.append(np.arange(losses[i].shape[0])*loss_freq)
        accbatch.append(np.arange(val_accs[i].shape[0])*acc_freq)

    figloss, axloss = plt.subplots(1,1)
    for i, fname in enumerate(filenames):
        axloss.plot(lossbatch[i], losses[i], label=labels[i])
    axloss.legend()
    axloss.set_title("Neural Stack, Adam, 50b, lr=0.001, Cross-Entropy Loss")
    axloss.set_ylabel("cross entropy loss")
    axloss.set_xlabel("batch number")
    figloss.savefig('neural-stack/plots/strangestack_loss.pdf', bbox_inches='tight')

    figval, axval = plt.subplots(1,1)
    for i, fname in enumerate(filenames):
        axval.plot(accbatch[i], val_accs[i], label=labels[i])
    axval.legend()
    axval.set_title("Neural Stack, Adam, 50b, lr=0.001, Validation Accuracy")
    axval.set_ylabel("fine accuracy")
    axval.set_xlabel("batch number")
    figval.savefig('neural-stack/plots/strangestack_val.pdf', bbox_inches='tight')

    figtest, axtest = plt.subplots(1,1)
    for i, fname in enumerate(filenames):
        axtest.plot(accbatch[i], test_accs[i], label=labels[i])
    axtest.legend()
    axtest.set_title("Neural Stack, Adam, 50b, lr=0.001, Test Accuracy")
    axtest.set_ylabel("fine accuracy")
    axtest.set_xlabel("batch number")
    figtest.savefig('neural-stack/plots/strangestack_test.pdf', bbox_inches='tight')

    plt.show()

