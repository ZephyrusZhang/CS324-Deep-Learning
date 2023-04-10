from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
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

        self.Whx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Whh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Why = nn.Linear(hidden_dim, output_dim, bias=True)
        # self.bh = Parameter(torch.zeros(hidden_dim))
        # self.bo = Parameter(torch.zeros(output_dim))

    def forward(self, x):
        x = x.view(self.batch_size, self.seq_length, self.input_dim)
        h = torch.zeros(self.batch_size, self.hidden_dim).cuda()
        for t in range(self.seq_length):
            xt = x[:, t, :]
            h = torch.tanh(self.Whx(xt) + self.Whh(h))
        o = self.Why(h)
        y = softmax(o, dim=1)
        return y

    # add more methods here if needed
