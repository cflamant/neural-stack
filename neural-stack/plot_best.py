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
    acc_freq = [10000,10000,1000,1000,1000,1000]

    filenames = ['neural-stack/models/longrnn_256h_4l_50b_Adam_1e-4',
                 'neural-stack/models/vlonglstm_256h_2l_50b_Adam_1e-3',
                 'neural-stack/models/manystack_64h_64e_1l_50b_Adam_1e-3_2'
                ]

    labels = ["RNN 256h 4l",
              "LSTM 256h 2l",
              "Neural Stack 64h 64e 1l"
              ]
    losses, val_accs, test_accs = [], [], []
    lossbatch, accbatch = [], []
    for i,fname in enumerate(filenames):
        losses.append(np.load(fname + '_loss.npy'))
        val_accs.append(np.load(fname + '_valaccs.npy'))
        test_accs.append(np.load(fname + '_testaccs.npy'))
        lossbatch.append(np.arange(losses[i].shape[0])*loss_freq)
        accbatch.append(np.arange(val_accs[i].shape[0])*acc_freq[i])

    figloss, axloss = plt.subplots(1,1)
    for i, fname in enumerate(filenames):
        axloss.semilogx(lossbatch[i], losses[i], label=labels[i])
    axloss.legend()
    axloss.set_title("Best Models, Cross-Entropy Loss")
    axloss.set_ylabel("cross entropy loss")
    axloss.set_xlabel("batch number")
    axloss.set_xlim(1000,500000)
    figloss.savefig('neural-stack/plots/best_loss.pdf', bbox_inches='tight')

    figval, axval = plt.subplots(1,1)
    for i, fname in enumerate(filenames):
        axval.semilogx(accbatch[i], val_accs[i], label=labels[i])
    axval.legend()
    axval.set_title("Best Models, Validation Accuracy")
    axval.set_ylabel("fine accuracy")
    axval.set_xlabel("batch number")
    axval.set_xlim(1000,500000)
    figval.savefig('neural-stack/plots/best_val.pdf', bbox_inches='tight')

    figtest, axtest = plt.subplots(1,1)
    for i, fname in enumerate(filenames):
        axtest.semilogx(accbatch[i], test_accs[i], label=labels[i])
    axtest.legend()
    axtest.set_title("Best Models, Test Accuracy")
    axtest.set_ylabel("fine accuracy")
    axtest.set_xlabel("batch number")
    axtest.set_xlim(1000,500000)
    figtest.savefig('neural-stack/plots/best_test.pdf', bbox_inches='tight')

    plt.show()

