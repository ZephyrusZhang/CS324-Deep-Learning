from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from torch.nn import Parameter
from torch.nn.functional import softmax


class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.Whx = Parameter(torch.randn(input_dim, hidden_dim))
        self.Whh = Parameter(torch.randn(hidden_dim, hidden_dim))
        self.bh = Parameter(torch.zeros(hidden_dim))
        self.Why = Parameter(torch.randn(hidden_dim, output_dim))
        self.bo = Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = x.view(self.batch_size, self.seq_length, self.input_dim)
        h = Parameter(torch.zeros(self.batch_size, self.hidden_dim)).cuda()
        for t in range(self.seq_length):
            xt = x[:, t, :]
            h = torch.tanh(xt @ self.Whx + h @ self.Whh + self.bh)
        y = h @ self.Why + self.bo
        y = softmax(y, dim=1)
        return y

    # add more methods here if needed
