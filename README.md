# neural-stack
Implementing DeepMind's Differentiable Neural Stack from scratch using PyTorch

Based on "Learning to Transduce with Unbounded Memory" by Edward Grefenstette, Karl Moritz Hermann, Mustafa Suleyman, Phil Blunsom. ([arXiv](https://arxiv.org/abs/1506.02516))

I implemented and trained a differentiable neural stack for an assignment in CS281: Advanced Machine Learning at Harvard in 2019, and was the only student in the class to successfully make it work.

## Description

A multi-layer vanilla RNN, LSTM, and LSTM with a neural stack are trained to reverse an input sequence of characters. However, strings in the training data have lengths between 8 and 64 characters, while the test set has strings of lengths 65 to 128. This crucial distinction between the training and test sets will gauge the generalizability of the network, and will show if the networks actually *learned* the concept of reversing a sequence.

As an example of the input and target data, suppose '$' is the start character, '|' is the separator, and '&' is the termination character. The input could be:

$A string to reverse|

and the expected output would be:

esrever ot gnirts A&

I tested standard recurrent neural networks (Elman RNN and LSTM) as a baseline for this task, and an LSTM connected to DeepMind's differentiable neural stack. The differentiable stack architecture can be used as a standard stack data structure, but it has inputs that can be controlled by a neural network. All the operations in the stack are differentiable, allowing for standard backpropagation techniques to train the neural network to use the stack in a way it finds useful, storing whatever type of encoded information in the data structure. 

![neural stack schematic](neural-stack/plots/neuralstack.png)
From [Learning to Transduce with Unbounded Memory](https://arxiv.org/abs/1506.02516)


