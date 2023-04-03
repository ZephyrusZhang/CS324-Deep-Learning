from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from cnn_model import CNN

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_EPOCHS_DEFAULT = 500
EVAL_FREQ_DEFAULT = 1
OPTIMIZER_DEFAULT = 'ADAM'

FLAGS = None


@torch.no_grad()
def accuracy(model, loader):
    correct, total = 0, 0
    for data in loader:
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return correct / total


@torch.no_grad()
def accuracy_loss(model, loader, criterion):
    running_loss = 0.0
    correct, total = 0, 0
    for data in loader:
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        running_loss += loss.item()
    return correct / total, running_loss / total


def train(opt, train_loader, test_loader):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    x_axis = []
    train_accuracy, train_loss, test_accuracy, test_loss = [], [], [], []
    cnn = CNN(3, 10).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=opt.learning_rate)

    print(cnn)
    for epoch in range(opt.max_steps):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            outputs = cnn(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % opt.eval_freq == 0:
            x_axis.append(epoch)
            train_accuracy.append(accuracy(cnn, train_loader))
            train_loss.append(running_loss / len(train_loader))
            acc, loss = accuracy_loss(cnn, test_loader, criterion)
            test_accuracy.append(acc)
            test_loss.append(loss)
            print(f'{datetime.datetime.now()}: '
                  f'Epoch {x_axis[-1]}\t'
                  f'Train Accuracy: {train_accuracy[-1]:.3f} \t Train Loss: {train_loss[-1]:.3f}\t'
                  f'Test Accuracy: {test_accuracy[-1]:.3f} \t Test Loss: {test_loss[-1]:.3f}')

    plt.plot(x_axis, train_accuracy, label='train accuracy')
    plt.plot(x_axis, train_loss, label='train loss')
    plt.plot(x_axis, test_accuracy, label='test accuracy')
    plt.plot(x_axis, test_loss, label='test loss')
    plt.legend()
    plt.show()


def main(opt):
    """
    Main function
    """
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)

    train(opt, train_loader, test_loader)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    # parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
    #                     help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS)
