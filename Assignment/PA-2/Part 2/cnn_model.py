from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn

from torch.nn import Linear, MaxPool2d, Softmax


class ConvBNReLU(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(ConvBNReLU, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      in_channels=in_channels,
                      out_channels=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)


class CNN(nn.Module):

    def __init__(self, n_channels, n_classes):
        """
        Initializes CNN object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=n_channels, out_channels=64),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=64, out_channels=128),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=128, out_channels=256),
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=256, out_channels=256),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=256, out_channels=512),
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=512, out_channels=512),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=512, out_channels=512),
            ConvBNReLU(kernel_size=3, stride=1, padding=1, in_channels=512, out_channels=512),
            MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.fc = nn.Sequential(
            Linear(in_features=512, out_features=n_classes),
            Softmax(dim=1)
        )

    def forward(self, x):
        """
        Performs forward pass of the input.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        x = self.conv_layers(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x
