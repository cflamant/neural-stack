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

    filenames = ['neural-stack/models/manystack_64h_64e_1l_50b_Adam_1e-3_1',
                 'neural-stack/models/manystack_64h_64e_1l_50b_Adam_1e-3_2',
                 'neural-stack/models/manystack_64h_64e_1l_50b_Adam_1e-3_3',
                 'neural-stack/models/manystack_64h_64e_1l_50b_Adam_1e-3_5',
                 'neural-stack/models/stack_64h_64e_1l_50b_Adam_1e-3_accfreq1000',
                 'neural-stack/models/stack_64h_64e_1l_50b_Adam_1e-3_accfreq1000_2'
                ]

    labels = ["run 1",
              "run 2",
              "run 3",
              "run 4",
              "run 5",
              "run 6"
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
    axloss.set_title("Neural Stack Consistency, Adam, 64h, 64e, 50b, Cross-Entropy Loss")
    axloss.set_ylabel("cross entropy loss")
    axloss.set_xlabel("batch number")
    figloss.savefig('neural-stack/plots/manystack_loss.pdf', bbox_inches='tight')

    figval, axval = plt.subplots(1,1)
    for i, fname in enumerate(filenames):
        axval.plot(accbatch[i], val_accs[i], label=labels[i])
    axval.legend()
    axval.set_title("Neural Stack Consistency, Adam, 64h, 64e, 50b, Validation Accuracy")
    axval.set_ylabel("fine accuracy")
    axval.set_xlabel("batch number")
    figval.savefig('neural-stack/plots/manystack_val.pdf', bbox_inches='tight')

    figtest, axtest = plt.subplots(1,1)
    for i, fname in enumerate(filenames):
        axtest.plot(accbatch[i], test_accs[i], label=labels[i])
    axtest.legend()
    axtest.set_title("Neural Stack Consistency, Adam, 64h, 64e, 50b, Test Accuracy")
    axtest.set_ylabel("fine accuracy")
    axtest.set_xlabel("batch number")
    figtest.savefig('neural-stack/plots/manystack_test.pdf', bbox_inches='tight')

    plt.show()

