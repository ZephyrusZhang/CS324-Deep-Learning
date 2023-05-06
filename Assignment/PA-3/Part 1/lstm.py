from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Initialization here ...
        self.Wgx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Wgh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Wix = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Wih = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Wfx = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Wfh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Wox = nn.Linear(input_dim, hidden_dim, bias=False)
        self.Woh = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Wph = nn.Linear(hidden_dim, output_dim, bias=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Implementation here ...
        batch_size, seq_length = x.size(0), x.size(1)
        x = x.view(batch_size, seq_length, self.input_dim)
        h_t = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t = torch.zeros(batch_size, self.hidden_dim).cuda()
        for t in range(seq_length):
            xt = x[:, t, :]
            g_t = self.tanh(self.Wgx(xt) + self.Wgh(h_t))
            i_t = self.sigmoid(self.Wix(xt) + self.Wih(h_t))
            f_t = self.sigmoid(self.Wfx(xt) + self.Wfh(h_t))
            o_t = self.sigmoid(self.Wox(xt) + self.Woh(h_t))
            c_t = g_t * i_t + c_t * f_t
            h_t = self.tanh(c_t) * o_t
        p_t = self.Wph(h_t)
        y_t = self.softmax(p_t)
        return y_t
