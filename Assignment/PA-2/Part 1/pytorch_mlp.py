from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super().__init__()
        self.model = nn.Sequential()
        units = [n_inputs] + n_hidden + [n_classes]
        for i in range(len(n_hidden)):
            self.model.append(nn.Linear(units[i], units[i + 1]))
            self.model.append(nn.ReLU())
        self.model.append(nn.Linear(units[-2], units[-1]))
        self.model.append(nn.Softmax(dim=1))

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        return self.model(x)
