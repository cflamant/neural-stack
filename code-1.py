"""
Author: Cedric Flamant
CS 281 Set 3
Exercise 1 and 2, RNN, LSTM, LSTM+NeuralStack to reverse strings
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import sys
import os
import tqdm  # so we can better tell how long training will take


class Reverser():
    """ A class that holds a backend for reversing strings, whether it be a vanilla RNN,
    LSTM, or LSTM with a neural stack. It also handles training.
    """

    def __init__(self, backend, **kwargs):
        """Initialize this container with a backend

        Parameters
        ----------
        backend : RNN_Reverser, LSTM_Reverser, or NeuralStack_Reverser object
            The backend neural architecture to use for the reversing task.
        **kwargs :
            See set_hyperparams()

        Returns
        -------
        None
        """
        self.use_cuda = backend.use_cuda
        if self.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print("CUDA is not available.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.backend = backend
        self.backend.to(self.device)

        # Get symbol dictionary size
        self.in_dim = self.backend.in_dim
        # For convenience, define integers of start symbol, separator, and terminate
        self.strt = 0
        self.sep = 1
        self.term = 2

        p = self.set_hyperparams(**kwargs)
        self.p = p

        # set optimizer
        opt_dic = {
                    'SGD': torch.optim.Adam(backend.parameters(), lr=p['lr']),
                    'Adam': torch.optim.Adam(backend.parameters(), lr=p['lr']),
                    'RMSprop': torch.optim.RMSprop(backend.parameters(), lr=p['lr']),
                  }
        self.optimizer = opt_dic[p['optimizer']]

        # Negative Log Likelihood Loss for training
        self.NLLL = nn.NLLLoss()
        #self.data = []
        ## TODO
        #for i in range(50):
        #    seq_len = np.random.randint(1,5)
        #    #seq_len = 4
        #    self.data.append(torch.randint(3,self.in_dim,(seq_len,1), device=self.device))


    def set_hyperparams(self, lr=0.001,
                              optimizer='SGD',
                              **kwargs):
        """Initialize dictionary of hyperparameters for training.

        Parameters
        ----------
        lr : float
            Learning rate for the optimizer
        optimizer : String
            String corresponding to which torch.optim optimizer to use (SGD, Adam)

        Returns
        -------
        p : dictionary
            Dictionary containing hyperparameters by name

        """


        p = dict(lr=lr,
                 optimizer=optimizer)
        return p

    def generate_seqs(self, batch_size, seq_len):
        """Generate int-encoded sequences, both X input and Y output.

        Parameters
        ----------
        batch_size : int
            Size of batch
        seq_len : int
            Sequence length

        Returns
        -------
        Xint : LongTensor of dim (seq_len+2, batch_size)
            int-encoded input sequence, with start and separator character
        Yint : LongTensor of dim (seq_len+1, batch_size)
            int-encoded output sequence, with terminator character
        """
        # Set up tensor of int-encoded symbols for start, separator, and terminal chars
        strtsym = torch.full((1,batch_size), self.strt, dtype=torch.int64, device=self.device)
        sepsym = torch.full((1,batch_size), self.sep, dtype=torch.int64, device=self.device)
        termsym = torch.full((1,batch_size), self.term, dtype=torch.int64, device=self.device)

        Xint = torch.randint(3,self.in_dim,(seq_len,batch_size), device=self.device)
        #TODO
        #Xint = self.data[np.random.randint(len(self.data))]

        # Reverse the sequence
        Yint = torch.flip(Xint, [0])
        # Append start and separator symbols to X, int-encoded
        Xint = torch.cat((strtsym, Xint, sepsym), dim=0)
        # Append terminal symbol to Y, int-encoded.
        Yint = torch.cat((Yint, termsym), dim=0)

        return Xint, Yint


    def train(self, batch_size=1, num_batch=10, seq_lim=(8,64)):
        """Train the backend with generated sequences

        Parameters
        ----------
        batch_size : int
            Size of batch, default 1
        num_batch : int
            Number of batches to create and train on
        seq_lim : tuple of 2 int
            Shortest sequence length and longest possible sequence length

        Returns
        -------
        train_loss : list
            List of cross-entropy loss 

        """
        
        self.backend.train()  # set in training mode

        train_loss = []
        with tqdm.trange(num_batch) as batches:
            for b in batches:
                # Pick a sequence length
                seq_len = np.random.randint(seq_lim[0],seq_lim[1]+1)
                # Create a batch of int-encoded symbol sequences
                Xint, Yint = self.generate_seqs(batch_size, seq_len)
                # Convert to one-hot
                X = F.one_hot(Xint, num_classes=self.in_dim).float()
                Y = F.one_hot(Yint, num_classes=self.in_dim).float()

                # Call neural backend forward pass
                Ypred = self.backend(X, Y=Y)

                self.optimizer.zero_grad()

                # Compute cross-entropy loss
                # TODO
                #print(torch.exp(Ypred).flatten(0,1))
                #print(torch.argmax(Ypred.flatten(0,1), dim=-1))
                #print(Yint.flatten())
                loss = self.NLLL(Ypred.flatten(0,1), Yint.flatten())

                loss.backward()

                #TODO
                #torch.nn.utils.clip_grad_norm_(self.backend.parameters(),1)
                self.optimizer.step()

                # record loss
                loss_val = loss.cpu().item()
                train_loss.append(loss_val)
                batches.set_postfix(loss=f'{loss_val:.2e}')

        return train_loss

    def reverse(self, Xint):
        """Given Xint in int-encoding, return Yint in int-encoding

        Parameters
        ----------
        Xint : LongTensor of dim (seq_len+2, 1)
            int-encoded input sequence, with start and separator character

        Returns
        -------
        Yint : LongTensor of dim (*, 1)
            int-encoded output sequence, with terminator character
        """
        # Set architecture in eval mode
        self.backend.eval()
        # Convert to one-hot
        X = F.one_hot(Xint, num_classes=self.in_dim).float()

        # Call neural backend forward pass
        Ypred = self.backend(X)
        Yint = torch.argmax(Ypred, dim=-1)
        return Yint



class RNN_Reverser(nn.Module):
    """A vanilla RNN seq2seq network for reversing strings. Has two main RNN parts, 
    an encoder RNN and a decoder RNN. The RNNs use an Elman architecture.
    """

    def __init__(self, use_cuda=False, **kwargs):
        """Initialize the network. Various hyperparameters are accepted, see the function
        set_hyperparams() for descriptions
        """
        super(RNN_Reverser, self).__init__()

        self.p = self.set_hyperparams(**kwargs)
        self.use_cuda = use_cuda

        if self.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print("Backend: CUDA is not available.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.in_dim = self.p['in_dim']
        self.out_dim = self.p['out_dim']
        self.hidden_dim = self.p['hidden_dim']
        self.num_layers = self.p['num_layers']

        self.max_len = self.p['max_len']

        # For convenience, define indices of start symbol, separator, and terminate
        self.strt = 0
        self.sep = 1
        self.term = 2
        
        # Set up layers
        # We'll use an RNN layer for the encoder, since the input has known length and is
        # more efficiently coded.
        self.encoder = nn.RNN(input_size=self.in_dim,
                              hidden_size=self.hidden_dim,
                              num_layers=self.num_layers)
        # The decoder will just consist of RNNCells since we will need to do a conditional
        # check to know when to stop outputting (when it produces a terminate symbol).
        self.decoder = nn.ModuleList()
        self.decoder.append(nn.RNNCell(input_size=self.in_dim, hidden_size=self.hidden_dim))
        for i in range(1, self.num_layers):
            self.decoder.append(nn.RNNCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim))
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=self.out_dim)

        # Log softmax (for use with NLLLoss error layer)
        self.logsoftmax = nn.LogSoftmax(dim=-1)



    def set_hyperparams(self, hidden_dim=16,
                              in_dim=131,
                              out_dim=131,
                              num_layers=1,
                              max_len=128,
                              **kwargs):
        """Initialize dictionary of hyperparameters for model.

        Parameters
        ----------
        hidden_dim : int
            Size of the hidden state
        in_dim : int
            Size of the input dimension. Should be equal to the number of possible 
            characters in the string to be reversed, plus 3 for the starting symbol,
            separator, and terminal symbol.
        out_dim : int
            Size of the output dimension. Should be equal to the number of possible
            characters in the string to be reversed, plus 3 for the starting symbol,
            separator, and terminal symbol.
        num_layers : int
            Number of RNN layers to stack.
        max_len : int
            Maximum length of output. Can be made arbitrarily large if long strings
            are expected. Just used to avoid infinite output.

        Returns
        -------
        p : dictionary
            Dictionary containing hyperparameters by name

        """

        p = dict(hidden_dim=hidden_dim,
                      in_dim=in_dim,
                      out_dim=out_dim,
                      num_layers=num_layers,
                      max_len=max_len)
        return p


    def forward(self, X, Y=None):
        """Forward propagation of the module. Takes in a sequence of one-hot-encoded
        vectors each representing one of the 128 characters, returns a sequence of
        log-probabilities of each symbol (logSoftmax on last layer).

        Parameters
        ----------
        X : tensor of dimension (seq_len, minibatch, in_dim)
            Input one-hots corresponding to sequence of characters. Note that all sequences
            in the minibatch must have the same length (so the separator appears in the same
            place)
        Y : tensor of dimension (seq_len, minibatch, in_dim)
            Output of one-hots corresponding to correct target sequence of characters. Only
            used if in training.

        Returns
        -------
        Ypred : tensor of dimension (*, minibatch, out_dim)
            If training:
            Returns a sequence of log-probabilities of each symbol (logSoftmax on last layer)
            Length is variable, and determined by the RNN itself based on when the terminal
            symbol is output (i.e. has the highest probability)
        """

        T_in = X.shape[0]
        batch_size = X.shape[1]
        in_dim = X.shape[2]
        if not self.training and batch_size > 1:
            # Not an intended use case
            raise IndexError("Batch size can only be 1 when model is in evaluation mode")
        if self.training and Y is None:
            raise ValueError("Target Y required in training mode.")

        # Encode through the entire input sequence excluding the separator.
        enc_outputs, hn = self.encoder(X[:-1,:,:])

        # initialize output tensor
        if self.training:
            # No need to let it output further than the loss will compare, which
            # is the length k + 1 for the terminator. 
            Ypred = torch.zeros((T_in-1,batch_size,in_dim), device=self.device)
        else:
            # Potentially can output indefinitely, so we give it a maximum length to
            # avoid it not stopping. However, we don't tell it how long the target is.
            Ypred = torch.zeros((self.max_len+1, batch_size, in_dim), device=self.device)

        T_out = Ypred.shape[0]

        ##################################################################
        ## Decode, but keep going until the stop character is produced. ##
        ##################################################################

        # Set up next hidden state
        hnp = []
        # Start with the separator character, which should be the last char in the input
        hnp.append(self.decoder[0](X[-1,:,:], hn[0]))
        for i in range(1,self.num_layers):  # If more than one recurrent layer
            hnp.append(self.decoder[i](hnp[i-1], hn[i]))
        hn = hnp
        dec_output = self.linear(hn[-1])
        # Now pass it through a logsoftmax
        Ypred[0,:,:] = self.logsoftmax(dec_output)

        t = 1
        # Only relevant if not training 
        if not self.training:
            # Stop if terminal symbol outputted (highest probability)
            stop = (torch.argmax(Ypred[0,0,:]).item() == self.term)
        else:
            stop = False
        while t < T_out and not stop:
            hnp = []
            if self.training:
                hnp.append(self.decoder[0](Y[t-1,:,:], hn[0]))
                for i in range(1, self.num_layers):  # if more than one reccurent layer
                    hnp.append(self.decoder[i](hnp[i-1], hn[i]))
            else:
                # Use most likely previous output as input (i.e. "greedy" approach
                # as opposed to beam search:
                # https://guillaumegenthial.github.io/sequence-to-sequence.html )
                Yprevint = torch.argmax(Ypred[t-1,:,:], dim=-1)
                Yprev = F.one_hot(Yprevint, num_classes=self.in_dim).float()
                hnp.append(self.decoder[0](Yprev, hn[0]))
                for i in range(1, self.num_layers):
                    hnp.append(self.decoder[i](hnp[i-1], hn[i]))
            hn = hnp
            dec_output = self.linear(hn[-1])
            # Now pass it through a logsoftmax
            Ypred[t,:,:] = self.logsoftmax(dec_output)
            if not self.training:
                # Stop if terminal symbol outputted (highest probability)
                stop = (torch.argmax(Ypred[t,0,:]).cpu().item() == self.term)
            t += 1

        return Ypred[:t,:,:]


class LSTM_Reverser(nn.Module):
    """An LSTM seq2seq network for reversing strings. Has two main LSTM parts, 
    an encoder LSTM and a decoder LSTM. 
    """

    def __init__(self, use_cuda=False, **kwargs):
        """Initialize the network. Various hyperparameters are accepted, see the function
        set_hyperparams() for descriptions
        """
        super(LSTM_Reverser, self).__init__()

        self.p = self.set_hyperparams(**kwargs)
        self.use_cuda = use_cuda

        if self.use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                print("Backend: CUDA is not available.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        self.in_dim = self.p['in_dim']
        self.out_dim = self.p['out_dim']
        self.hidden_dim = self.p['hidden_dim']
        self.num_layers = self.p['num_layers']

        self.max_len = self.p['max_len']

        # For convenience, define indices of start symbol, separator, and terminate
        self.strt = 0
        self.sep = 1
        self.term = 2
        
        # Set up layers
        # We'll use a LSTM layer for the encoder, since the input has known length and is
        # more efficiently coded.
        self.encoder = nn.LSTM(input_size=self.in_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.num_layers)
        # The decoder will just consist of LSTMCells since we will need to do a conditional
        # check to know when to stop outputting (when it produces a terminate symbol).
        self.decoder = nn.ModuleList()
        self.decoder.append(nn.LSTMCell(input_size=self.in_dim, hidden_size=self.hidden_dim))
        for i in range(1, self.num_layers):
            self.decoder.append(nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim))
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=self.out_dim)


        # Log softmax (for use with NLLLoss error layer)
        self.logsoftmax = nn.LogSoftmax(dim=-1)



    def set_hyperparams(self, hidden_dim=16,
                              in_dim=131,
                              out_dim=131,
                              num_layers=1,
                              max_len=128,
                              **kwargs):
        """Initialize dictionary of hyperparameters for model.

        Parameters
        ----------
        hidden_dim : int
            Size of the hidden state
        in_dim : int
            Size of the input dimension. Should be equal to the number of possible 
            characters in the string to be reversed, plus 3 for the starting symbol,
            separator, and terminal symbol.
        out_dim : int
            Size of the output dimension. Should be equal to the number of possible
            characters in the string to be reversed, plus 3 for the starting symbol,
            separator, and terminal symbol.
        num_layers : int
            Number of LSTM layers to stack.
        max_len : int
            Maximum length of output. Can be made arbitrarily large if long strings
            are expected. Just used to avoid infinite output.

        Returns
        -------
        p : dictionary
            Dictionary containing hyperparameters by name

        """

        p = dict(hidden_dim=hidden_dim,
                      in_dim=in_dim,
                      out_dim=out_dim,
                      num_layers=num_layers,
                      max_len=max_len)
        return p


    def forward(self, X, Y=None):
        """Forward propagation of the module. Takes in a sequence of one-hot-encoded
        vectors each representing one of the 128 characters, returns a sequence of
        log-probabilities of each symbol (logSoftmax on last layer).

        Parameters
        ----------
        X : tensor of dimension (seq_len, minibatch, in_dim)
            Input one-hots corresponding to sequence of characters. Note that all sequences
            in the minibatch must have the same length (so the separator appears in the same
            place)
        Y : tensor of dimension (seq_len, minibatch, in_dim)
            Output of one-hots corresponding to correct target sequence of characters. Only
            used if in training.

        Returns
        -------
        Ypred : tensor of dimension (*, minibatch, out_dim)
            If training:
            Returns a sequence of log-probabilities of each symbol (logSoftmax on last layer)
            Length is variable, and determined by the decoder itself based on when the terminal
            symbol is output (i.e. has the highest probability)
        """

        T_in = X.shape[0]
        batch_size = X.shape[1]
        in_dim = X.shape[2]
        if not self.training and batch_size > 1:
            # Not an intended use case
            raise IndexError("Batch size can only be 1 when model is in evaluation mode")
        if self.training and Y is None:
            raise ValueError("Target Y required in training mode.")

        # Encode through the entire input sequence excluding the separator.
        enc_outputs, (hn, cn) = self.encoder(X[:-1,:,:])

        # initialize output tensor
        if self.training:
            # No need to let it output further than the loss will compare, which
            # is the length k + 1 for the terminator. 
            Ypred = torch.zeros((Y.shape[0],batch_size,in_dim), device=self.device)
        else:
            # Potentially can output indefinitely, so we give it a maximum length to
            # avoid it not stopping. However, we don't tell it how long the target is.
            Ypred = torch.zeros((self.max_len+1, batch_size, in_dim), device=self.device)

        T_out = Ypred.shape[0]

        ##################################################################
        ## Decode, but keep going until the stop character is produced. ##
        ##################################################################

        # Set up next hidden state and cell state
        hnp = []
        cnp = []
        # Start with the separator character, which should be the last char in the input
        htemp, ctemp = self.decoder[0](X[-1,:,:], (hn[0],cn[0]))
        hnp.append(htemp)
        cnp.append(ctemp)
        for i in range(1,self.num_layers):  # If more than one recurrent layer
            htemp, ctemp = self.decoder[i](hnp[i-1], (hn[i],cn[i]))
            hnp.append(htemp)
            cnp.append(ctemp)
        hn = hnp
        cn = cnp
        dec_output = self.linear(hn[-1])
        # Now pass it through a logsoftmax
        Ypred[0,:,:] = self.logsoftmax(dec_output)

        t = 1
        # Only relevant if not training 
        if not self.training:
            # Stop if terminal symbol outputted (highest probability)
            stop = (torch.argmax(Ypred[0,0,:]).item() == self.term)
        else:
            stop = False
        while t < T_out and not stop:
            hnp = []
            cnp = []
            if self.training:
                htemp, ctemp = self.decoder[0](Y[t-1,:,:], (hn[0], cn[0]))
                hnp.append(htemp)
                cnp.append(ctemp)
                for i in range(1, self.num_layers):  # if more than one recurrent layer
                    htemp, ctemp = self.decoder[i](hnp[i-1], (hn[i], cn[i]))
                    hnp.append(htemp)
                    cnp.append(ctemp)
            else:
                # Use most likely previous output as input (i.e. "greedy" approach
                # as opposed to beam search:
                # https://guillaumegenthial.github.io/sequence-to-sequence.html )
                Yprevint = torch.argmax(Ypred[t-1,:,:], dim=-1)
                Yprev = F.one_hot(Yprevint, num_classes=self.in_dim).float()
                htemp, ctemp = self.decoder[0](Yprev, (hn[0], cn[0]))
                hnp.append(htemp)
                cnp.append(ctemp)
                for i in range(1, self.num_layers):
                    htemp, ctemp = self.decoder[i](hnp[i-1], (hn[i], cn[i]))
                    hnp.append(htemp)
                    cnp.append(ctemp)
            hn, cn = hnp, cnp
            dec_output = self.linear(hn[-1])
            # Now pass it through a logsoftmax
            Ypred[t,:,:] = self.logsoftmax(dec_output)
            if not self.training:
                # Stop if terminal symbol outputted (highest probability)
                stop = (torch.argmax(Ypred[t,0,:]).cpu().item() == self.term)
            t += 1

        return Ypred[:t,:,:]
###########################################################################################
# Begin main portion of script
###########################################################################################

if __name__ == "__main__":
    # Any random seed.
    np.random.seed(None)
    savefile = 'models/lstm_test_10i20h2l'
    overwrite = False

    #torch.autograd.set_detect_anomaly(True)
    #rnn = RNN_Reverser(use_cuda=True, hidden_dim=20, in_dim=10, out_dim=10, max_len=15, num_layers=2)
    lstm = LSTM_Reverser(use_cuda=True, hidden_dim=20, in_dim=10, out_dim=10, max_len=15, num_layers=2)
    #reverser = Reverser(rnn, optimizer='Adam', lr=0.0001)
    reverser = Reverser(lstm, optimizer='Adam', lr=0.0001)

    if os.path.isfile(savefile + '.pt') and not overwrite:
        reverser.backend.load_state_dict(torch.load(savefile + '.pt'))
    if os.path.isfile(savefile + '_loss.npy') and not overwrite:
        old_loss = np.load(savefile + '_loss.npy')
    else:
        old_loss = np.array([])

    lims = (4,12)
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



