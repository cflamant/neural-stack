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
import dataloader # Get training and validation data


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

        # Get validation and test data for accuracy scoring
        self.val_x, self.val_y, self.test_x, self.test_y = dataloader.get_data(self.device)


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

        # Reverse the sequence
        Yint = torch.flip(Xint, [0])
        # Append start and separator symbols to X, int-encoded
        Xint = torch.cat((strtsym, Xint, sepsym), dim=0)
        # Append terminal symbol to Y, int-encoded.
        Yint = torch.cat((Yint, termsym), dim=0)

        return Xint, Yint

    def accuracy_scores(self):
        """Calculate fine-grained accuracy score on validation and test sets

        Parameters
        ----------
        None

        Returns
        -------
        val_acc : float
            Accuracy score on validation set
        test_acc : float
            Accuracy score on test set

        """
        val_acc = 0.
        test_acc = 0.
        for i in range(len(self.val_x)):
            ypred = self.reverse(self.val_x[i]).flatten().cpu().numpy()
            num_correct = 0
            for j in range(min(self.val_y[i].shape[0],ypred.shape[0])):
                if self.val_y[i][j] == ypred[j]:
                    num_correct += 1
            val_acc += num_correct/float(self.val_y[i].shape[0])
        val_acc /= len(self.val_y)

        for i in range(len(self.test_x)):
            ypred = self.reverse(self.test_x[i]).flatten().cpu().numpy()
            num_correct = 0
            for j in range(min(self.test_y[i].shape[0],ypred.shape[0])):
                if self.test_y[i][j] == ypred[j]:
                    num_correct += 1
            test_acc += num_correct/float(self.test_y[i].shape[0])
        test_acc /= len(self.test_y)

        # Set back to training mode!
        self.backend.train()

        #TODO May be removed. Useful to keep tabs on progress
        print(f'val_acc = {val_acc}')
        print(f'test_acc = {test_acc}')

        return val_acc, test_acc



    def train(self, batch_size=1, num_batch=10, seq_lim=(8,64), loss_freq=100, acc_freq=10000, last=True):
        """Train the backend with generated sequences

        Parameters
        ----------
        batch_size : int
            Size of batch, default 1
        num_batch : int
            Number of batches to create and train on
        seq_lim : tuple of 2 int
            Shortest sequence length and longest possible sequence length
        loss_freq : int
            How often to save losses (every loss_freq batches). This is to
            prevent unnecessarily large loss lists from being created on long runs.
        acc_freq : int
            How often to compute validation and test accuracies (every
            acc_freq batches). Negative or 0 value skips accuracy calculation.
        last : bool
            Whether to compute the accuracy after the last iteration.

        Returns
        -------
        train_losses : List[float]
            List of cross-entropy loss 
        val_accs : List[float]
            List of validation set accuracy 
        test_accs : List[float]
            List of test set accuracy 

        """
        
        self.backend.train()  # set in training mode

        # don't compute accuracy if this is given
        no_acc_compute = (acc_freq <= 0)

        train_losses = []
        val_accs = []
        test_accs = []
        iternum = 0
        with tqdm.trange(num_batch) as batches:
            for b in batches:
                # Compute accuracy score on validation and test
                if not no_acc_compute and iternum % acc_freq == 0:
                    val_acc, test_acc = self.accuracy_scores()
                    val_accs.append(val_acc)
                    test_accs.append(test_acc)

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
                loss = self.NLLL(Ypred.flatten(0,1), Yint.flatten())

                loss.backward()

                #TODO Can add option for gradient clipping if it seems useful
                #torch.nn.utils.clip_grad_norm_(self.backend.parameters(),1)
                self.optimizer.step()

                # record loss
                loss_val = loss.cpu().item()
                if iternum % loss_freq == 0:
                    train_losses.append(loss_val)
                iternum += 1
                batches.set_postfix(loss=f'{loss_val:.2e}')
        if last:
            # Compute accuracy score on validation and test now that we're done
            if not no_acc_compute:
                val_acc, test_acc = self.accuracy_scores()
                val_accs.append(val_acc)
                test_accs.append(test_acc)

        return train_losses, val_accs, test_accs

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
        enc_outputs, hn_all = self.encoder(X[:-1,:,:])
        # Chunk tensor into list for hidden state of each layer
        hn = torch.chunk(hn_all.view(-1,self.hidden_dim),self.num_layers)

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
        enc_outputs, (hn_all, cn_all) = self.encoder(X[:-1,:,:])
        # Chunk tensor into list for hidden/cell state of each layer
        hn = torch.chunk(hn_all.view(-1,self.hidden_dim),self.num_layers)
        cn = torch.chunk(cn_all.view(-1,self.hidden_dim),self.num_layers)

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
        hnp, cnp = [], []
        # Start with the separator character, which should be the last char in the input
        htemp, ctemp = self.decoder[0](X[-1,:,:], (hn[0],cn[0]))
        hnp.append(htemp)
        cnp.append(ctemp)
        for i in range(1,self.num_layers):  # If more than one recurrent layer
            htemp, ctemp = self.decoder[i](hnp[i-1], (hn[i],cn[i]))
            hnp.append(htemp)
            cnp.append(ctemp)
        hn, cn = hnp, cnp
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
            hnp, cnp = [], []
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


class NeuralStack(nn.Module):
    """A neural stack (differentiable stack) as described in https://arxiv.org/pdf/1506.02516.pdf. 
    This does not include the controller unit.
    """

    def __init__(self):
        """Initialize the neural stack.
        All dimensions are inferred from what is passed to it.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        super(NeuralStack, self).__init__()


    def forward(self, V, s, d, u, v):
        """Forward method for the stack. Initiates one step in time.
        Note that it does not keep track of dimensions, so the user is
        responsible for sending the right shape of data so the output shapes
        can be inferred.

        Parameters
        ----------
        V : List[torch.Tensor of dim (batch_size,m)]
            Values list containing t-1 tensors where t is the current time
            and m is the size of the embedding
        s : List[torch.Tensor of dim (batch_size,1)]
            List of strengths corresponding to the vectors in V.
        d : torch.Tensor of dim (batch_size,1)
            Fraction to push
        u : torch.Tensor of dim (batch_size,1)
            Fraction to pop
        v : torch.Tensor of dim (batch_size,m)
            Value to potentially add to the stack.

        Returns
        -------
        Vn : List[torch.Tensor of dim (batch_size,m)]
            Values list containing t tensors where t is the current time. These
            are the next state's stack values.
        sn : List[torch.Tensor of dim (batch_size,1)]
            List of strengths corresponding to the vectors in Vn.
        r : torch.Tensor of dim (batch_size,m)
            Readout from stack. Has the interpretation of being a superposition of
            possible popped value vectors.

        """

        # Compute the current time
        t = len(s) + 1

        # Update V to make Vn
        V.append(v)
        Vn = V

        # Create new strength list
        sn = []
        for i in range(t-1):
            totprev = torch.zeros_like(d)
            for j in range(i+1,t-1):
                totprev += s[j]
            inside = F.relu(u - totprev)
            sn.append(F.relu(s[i] - inside))
        sn.append(d)

        # Create readout vector
        r = torch.zeros_like(v)
        for i in range(t):
            tots = torch.zeros_like(d)
            for j in range(i+1,t):
                tots += sn[j]
            inside = F.relu(1. - tots)
            r += torch.min(sn[i],inside) * Vn[i]

        return Vn, sn, r


class LSTMandStack_Reverser(nn.Module):
    """An LSTM equipped with a neural stack, seq2seq network for reversing strings.
    Has two main LSTM parts, an encoder LSTM and a decoder LSTM. Both interface with
    the same stack.
    """

    def __init__(self, use_cuda=False, **kwargs):
        """Initialize the network. Various hyperparameters are accepted, see the function
        set_hyperparams() for descriptions
        """
        super(LSTMandStack_Reverser, self).__init__()

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
        self.embedding_dim = self.p['embedding_dim']
        self.num_layers = self.p['num_layers']

        self.max_len = self.p['max_len']

        # For convenience, define indices of start symbol, separator, and terminate
        self.strt = 0
        self.sep = 1
        self.term = 2
        
        # Set up layers
        # The encoder will consist of LSTMCells so we can interface easily with the neural
        # stack after every step.
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.LSTMCell(input_size=self.in_dim+self.embedding_dim, hidden_size=self.hidden_dim))
        for i in range(1, self.num_layers):
            self.encoder.append(nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim))

        # The decoder will consist of LSTMCells since we will need to do a conditional
        # check to know when to stop outputting (when it produces a terminate symbol).
        self.decoder = nn.ModuleList()
        self.decoder.append(nn.LSTMCell(input_size=self.in_dim+self.embedding_dim, hidden_size=self.hidden_dim))
        for i in range(1, self.num_layers):
            self.decoder.append(nn.LSTMCell(input_size=self.hidden_dim, hidden_size=self.hidden_dim))
        self.linear = nn.Linear(in_features=self.hidden_dim, out_features=self.out_dim)

        # Initialize the neural stack
        self.nstack = NeuralStack()

        # Initialize projection matrices. Use Xavier initialization
        Wd = torch.randn((1,self.hidden_dim), device=self.device)
        nn.init.xavier_normal_(Wd)
        self.Wd = torch.nn.Parameter(data=Wd, requires_grad=True)
        self.bd = torch.nn.Parameter(data=torch.tensor(0., device=self.device), requires_grad=True)

        Wu = torch.randn((1,self.hidden_dim), device=self.device)
        nn.init.xavier_normal_(Wu)
        self.Wu = torch.nn.Parameter(data=Wu, requires_grad=True)
        self.bu = torch.nn.Parameter(data=torch.tensor(0., device=self.device), requires_grad=True)

        Wv = torch.randn((self.embedding_dim, self.hidden_dim), device=self.device)
        nn.init.xavier_normal_(Wv)
        self.Wv = torch.nn.Parameter(data=Wv, requires_grad=True)
        self.bv = torch.nn.Parameter(data=torch.zeros(self.embedding_dim, device=self.device),
                                     requires_grad=True)

        Wo = torch.randn((self.hidden_dim, self.hidden_dim), device=self.device)
        nn.init.xavier_normal_(Wo)
        self.Wo = torch.nn.Parameter(data=Wo, requires_grad=True)
        self.bo = torch.nn.Parameter(data=torch.zeros(self.hidden_dim, device=self.device),
                                     requires_grad=True)

        # Initialize hidden state (treated as parameter to be optimized)
        self.h0 = []
        self.c0 = []
        for l in range(self.num_layers):
            hh = torch.nn.Parameter(data=torch.randn(self.hidden_dim, device=self.device),
                                    requires_grad=True)
            cc = torch.nn.Parameter(data=torch.randn(self.hidden_dim, device=self.device),
                                    requires_grad=True)
            self.h0.append(hh)
            self.c0.append(cc)
        # Log softmax (for use with NLLLoss error layer)
        self.logsoftmax = nn.LogSoftmax(dim=-1)


    def set_hyperparams(self, hidden_dim=16,
                              in_dim=131,
                              out_dim=131,
                              embedding_dim=16,
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
        embedding_dim : int
            Size of the embedding for the neural stack (dimension of value vectors)
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
                      embedding_dim=embedding_dim,
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

        ##################################################################
        ## Encode until separator is hit                                ##
        ##################################################################
        T_in = X.shape[0]
        hn, cn = [], []
        for i in range(self.num_layers):
            hn.append(self.h0[i].expand(batch_size,-1))
            cn.append(self.c0[i].expand(batch_size,-1))
        # Initialize empty V, s, r
        V = [torch.zeros((batch_size,self.embedding_dim),device=self.device)]
        s = [torch.zeros((batch_size,1), device=self.device)]
        r = torch.zeros((batch_size,self.embedding_dim),device=self.device)
        for t in range(T_in-1):
            # Set up next hidden state and cell state
            hnp, cnp = [], []
            # concat read state to input
            layer1in = torch.cat([X[t,:,:], r], dim=-1)
            htemp, ctemp = self.encoder[0](layer1in, (hn[0], cn[0]))
            hnp.append(htemp)
            cnp.append(ctemp)
            for i in range(1, self.num_layers):  # if more than one recurrent layer
                htemp, ctemp = self.encoder[i](hnp[i-1], (hn[i], cn[i]))
                hnp.append(htemp)
                cnp.append(ctemp)
            hn, cn = hnp, cnp
            d = torch.sigmoid(torch.matmul(self.Wd,hn[-1].unsqueeze(-1)).squeeze(-1) + self.bd)
            u = torch.sigmoid(torch.matmul(self.Wu,hn[-1].unsqueeze(-1)).squeeze(-1) + self.bu)
            v = torch.tanh(torch.matmul(self.Wv,hn[-1].unsqueeze(-1)).squeeze(-1) + self.bv)
            # interface with stack
            V,s,r = self.nstack(V,s,d,u,v)

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
        hnp, cnp = [], []
        # Start with the separator character, which should be the last char in the input
        # concat read state to input
        layer1in = torch.cat([X[-1,:,:], r], dim=-1)
        htemp, ctemp = self.decoder[0](layer1in, (hn[0],cn[0]))
        hnp.append(htemp)
        cnp.append(ctemp)
        for i in range(1,self.num_layers):  # If more than one recurrent layer
            htemp, ctemp = self.decoder[i](hnp[i-1], (hn[i],cn[i]))
            hnp.append(htemp)
            cnp.append(ctemp)
        hn, cn = hnp, cnp
        d = torch.sigmoid(torch.matmul(self.Wd,hn[-1].unsqueeze(-1)).squeeze(-1) + self.bd)
        u = torch.sigmoid(torch.matmul(self.Wu,hn[-1].unsqueeze(-1)).squeeze(-1) + self.bu)
        v = torch.tanh(torch.matmul(self.Wv,hn[-1].unsqueeze(-1)).squeeze(-1) + self.bv)
        o = torch.tanh(torch.matmul(self.Wo,hn[-1].unsqueeze(-1)).squeeze(-1) + self.bo)
        # interface with stack
        V,s,r = self.nstack(V,s,d,u,v)
        dec_output = self.linear(o)
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
            hnp, cnp = [], []
            if self.training:
                layer1in = torch.cat([Y[t-1,:,:], r], dim=-1)
                htemp, ctemp = self.decoder[0](layer1in, (hn[0], cn[0]))
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
                layer1in = torch.cat([Yprev, r], dim=-1)
                htemp, ctemp = self.decoder[0](layer1in, (hn[0], cn[0]))
                hnp.append(htemp)
                cnp.append(ctemp)
                for i in range(1, self.num_layers):
                    htemp, ctemp = self.decoder[i](hnp[i-1], (hn[i], cn[i]))
                    hnp.append(htemp)
                    cnp.append(ctemp)
            hn, cn = hnp, cnp
            d = torch.sigmoid(torch.matmul(self.Wd,hn[-1].unsqueeze(-1)).squeeze(-1) + self.bd)
            u = torch.sigmoid(torch.matmul(self.Wu,hn[-1].unsqueeze(-1)).squeeze(-1) + self.bu)
            v = torch.tanh(torch.matmul(self.Wv,hn[-1].unsqueeze(-1)).squeeze(-1) + self.bv)
            o = torch.tanh(torch.matmul(self.Wo,hn[-1].unsqueeze(-1)).squeeze(-1) + self.bo)
            # interface with stack
            V,s,r = self.nstack(V,s,d,u,v)
            dec_output = self.linear(o)
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

# This section is mainly for quick testing of code. Real hyperparameter search runs are
# performed in accompanying scripts.

if __name__ == "__main__":
    # Any random seed.
    np.random.seed(None)
    testit = 0

    if testit == 0:
        savefile = 'models/stack_test_131i256h1l'
        overwrite = False

        #torch.autograd.set_detect_anomaly(True)
        #rnn = RNN_Reverser(use_cuda=True, hidden_dim=20, in_dim=10, out_dim=10, max_len=15, num_layers=2)
        #lstm = LSTM_Reverser(use_cuda=True, hidden_dim=256, in_dim=131, out_dim=131, max_len=128, num_layers=1)
        stack = LSTMandStack_Reverser(use_cuda=True, hidden_dim=256, in_dim=131, out_dim=131, embedding_dim=64, max_len=128, num_layers=1)
        #reverser = Reverser(rnn, optimizer='Adam', lr=0.0001)
        #reverser = Reverser(lstm, optimizer='Adam', lr=0.0001)
        reverser = Reverser(stack, optimizer='Adam', lr=0.0001)

        if os.path.isfile(savefile + '.pt') and not overwrite:
            reverser.backend.load_state_dict(torch.load(savefile + '.pt', device=self.device))
        if os.path.isfile(savefile + '_loss.npy') and not overwrite:
            old_loss = np.load(savefile + '_loss.npy')
        else:
            old_loss = np.array([])

        lims = (8,64)
        losses, val_accs, train_accs = reverser.train(batch_size=50, num_batch=200000, seq_lim=lims, acc_freq=-1)

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
    elif testit == 1:
        # test out neural stack.
        nstack = NeuralStack()
        batch_size = 5
        m = 3
        v1 = torch.ones((batch_size,m))
        v2 = torch.ones((batch_size,m)) * 2
        v3 = torch.ones((batch_size,m)) * 3

        u1 = torch.zeros((batch_size,1))
        u2 = torch.ones((batch_size,1)) * 0.1
        u3 = torch.ones((batch_size,1)) * 0.9

        d1 = torch.ones((batch_size,1)) * 0.8
        d2 = torch.ones((batch_size,1)) * 0.5
        d3 = torch.ones((batch_size,1)) * 0.9

        V = []
        s = []

        V, s, r = nstack(V,s,d1,u1,v1)
        print(r)
        V, s, r = nstack(V,s,d2,u2,v2)
        print(r)
        V, s, r = nstack(V,s,d3,u3,v3)
        print(r)



