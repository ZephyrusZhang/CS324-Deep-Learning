from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import PalindromeDataset
from lstm import LSTM
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model, loader, criterion):
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
    return correct / total, running_loss / len(loader)


def train(train_loader: DataLoader, test_loader: DataLoader):
    x_axis = []
    train_accuracy, train_loss, test_accuracy, test_loss = [], [], [], []

    # Initialize the model that we are going to use
    model = LSTM(
        OPT.input_dim,
        OPT.num_hidden,
        OPT.num_classes
    ).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=OPT.learning_rate)

    for epoch in range(OPT.max_epoch):
        running_loss = 0.0
        correct, total = 0, 0
        total_time = 0.0
        for step, (batch_inputs, batch_targets) in enumerate(train_loader):
            start_time = time.time()

            # the following line is to deal with exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=OPT.max_norm)

            # Add more code here ...
            batch_inputs, batch_targets = batch_inputs.cuda(), batch_targets.cuda()

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if epoch % OPT.eval_freq == 0:
                _, predicted = torch.max(outputs.data, dim=1)
                total += batch_targets.size(0)
                correct += (predicted == batch_targets).sum().item()

            total_time += time.time() - start_time

        if epoch % OPT.eval_freq == 0:
            x_axis.append(epoch)
            train_accuracy.append(correct / total)
            train_loss.append(running_loss / len(train_loader))
            acc, loss = evaluate(model, test_loader, criterion)
            test_accuracy.append(acc)
            test_loss.append(loss)

            if not OPT.quiet:
                print(f'Epoch {x_axis[-1]}\n'
                      f'Train Accuracy: {train_accuracy[-1]:.3f} \t Train Loss: {train_loss[-1]:.3f}\t'
                      f'Test Accuracy: {test_accuracy[-1]:.3f} \t Test Loss: {test_loss[-1]:.3f}\t'
                      f'Avg Time Cost: {total_time / (epoch + 1):.3f}s')

    print('Done training.')

    plt.plot(x_axis, train_accuracy, label='train accuracy')
    plt.plot(x_axis, train_loss, label='train loss')
    plt.plot(x_axis, test_accuracy, label='test accuracy')
    plt.plot(x_axis, test_loss, label='test loss')
    plt.legend()
    plt.title(f'seq_length = {OPT.input_length}')
    plt.show()


def main():
    train_dataset = PalindromeDataset(OPT.input_length + 1, int(OPT.data_size * OPT.portion_train))
    train_loader = DataLoader(train_dataset, OPT.batch_size)
    test_dataset = PalindromeDataset(OPT.input_length + 1, OPT.data_size - int(OPT.data_size * OPT.portion_train))
    test_loader = DataLoader(test_dataset, OPT.batch_size)

    train(train_loader, test_loader)


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=10,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--data_size', type=int, default=100000,
                        help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8,
                        help='Portion of the total dataset used for training')
    parser.add_argument('--max_epoch', type=int, default=100,
                        help='Number of epochs to run for')
    parser.add_argument('--quiet', action='store_true',
                        help='No stdout')
    parser.add_argument('--eval_freq', type=int, default=1)
    parser.add_argument('--max_norm', type=float, default=10.0)

    OPT = parser.parse_args()

    main()
