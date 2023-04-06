from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN


@torch.no_grad()
def evaluate(model, loader, criterion, n_samples):
    running_loss = 0.0
    correct, total = 0, 0
    for i, data in enumerate(loader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        if (i + 1) % n_samples == 0:
            break
    return correct / total, running_loss / total


def train(opt, train_loader, test_loader):
    x_axis = []
    train_accuracy, train_loss, test_accuracy, test_loss = [], [], [], []

    # Initialize the model that we are going to use
    model = VanillaRNN(
        seq_length=opt.input_length,
        input_dim=opt.input_dim,
        hidden_dim=opt.num_hidden,
        output_dim=opt.num_classes,
        batch_size=opt.batch_size
    ).cuda()

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=opt.learning_rate)

    running_loss = 0.0
    total_time = 0.0
    for step, (batch_inputs, batch_targets) in enumerate(train_loader):
        start_time = time.time()

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.max_norm)

        # Add more code here ...
        batch_inputs, batch_targets = batch_inputs.cuda(), batch_targets.cuda()

        # Add more code here ...
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_time += time.time() - start_time

        if step % opt.eval_freq == 0:
            _, predicted = torch.max(outputs.data, 1)
            x_axis.append(step)
            train_accuracy.append((predicted == batch_targets).sum().item() / batch_targets.size(0))
            train_loss.append(running_loss / (step + 1))
            acc, loss = evaluate(model, test_loader, criterion, opt.batch_size)
            test_accuracy.append(acc)
            test_loss.append(loss)
            print(f'Epoch {x_axis[-1]}\n'
                  f'Train Accuracy: {train_accuracy[-1]:.3f} \t Train Loss: {train_loss[-1]:.3f}\t'
                  f'Test Accuracy: {test_accuracy[-1]:.3f} \t Test Loss: {test_loss[-1]:.3f}\t'
                  f'Avg Time Cost: {total_time / (step + 1):.3f}s')

        if step == opt.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


def main(opt):
    train_dataset = PalindromeDataset(opt.input_length + 1)
    train_loader = DataLoader(train_dataset, opt.batch_size, num_workers=1)
    test_dataset = PalindromeDataset(opt.input_length + 1)
    test_loader = DataLoader(test_dataset, opt.batch_size, num_workers=1)

    train(opt, train_loader, test_loader)


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=5, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=5000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--eval_freq', type=int, default=10)

    config = parser.parse_args()
    # Train the model
    main(config)
