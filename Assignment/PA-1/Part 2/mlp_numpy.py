from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes, batch_size, lr: float):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """

        self.batch_size = batch_size
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.lr = lr

        self.units = [n_inputs] + n_hidden + [n_classes]
        self.linear = [Linear(self.units[i], self.units[i + 1]) for i in range(len(n_hidden) + 1)]
        self.relu = [ReLU() for _ in range(len(n_hidden))]
        self.softmax = SoftMax()

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out = x.copy()
        for i in range(len(self.n_hidden)):
            out = self.linear[i].forward(out)
            out = self.relu[i].forward(out)
        out = self.linear[-1].forward(out)
        out = self.softmax.forward(out)
        return out

    def backward(self, dout):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        dx = self.softmax.backward(dout)
        dx = self.linear[-1].backward(dx)
        for i in range(len(self.relu) - 1, -1, -1):
            dx = self.relu[i].backward(dx)
            dx = self.linear[i].backward(dx)

    def step(self):
        for linear in self.linear:
            linear.step(self.batch_size, self.lr)

    def zero_grad(self):
        for linear in self.linear:
            linear.zero_grad()
