from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import torch
import torch.nn as nn
import torch.optim as optim

from pytorch_mlp import MLP
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from torch import FloatTensor, LongTensor
from torch.utils.data import DataLoader, TensorDataset

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None


def make_loader(x_train, x_test, y_train, y_test, batch=1):
    train_dataset = TensorDataset(FloatTensor(x_train), LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch)

    test_dataset = TensorDataset(FloatTensor(x_test), LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(mlp, loader, criterion):
    running_loss = 0.0
    correct, total = 0, 0
    for i, data in enumerate(loader):
        inputs, labels = data
        outputs = mlp(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += np.around(predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        running_loss += loss.item()
    return correct / total, running_loss / len(loader)


def train(opt, x_train, x_test, y_train, y_test):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    x_axis = []
    train_accuracy, train_loss, test_accuracy, test_loss = [], [], [], []
    n_hidden = list(map(lambda ele: int(ele), opt.dnn_hidden_units.split(',')))
    train_loader, test_loader = make_loader(x_train, x_test, y_train, y_test)
    mlp = MLP(2, n_hidden, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mlp.parameters(), lr=opt.learning_rate)

    total_time = 0
    for epoch in range(opt.max_steps):
        running_loss = 0.0
        correct, total = 0, 0
        start_time = time.time()
        for data in train_loader:
            inputs, labels = data  # 获取此次训练的输入和标签
            optimizer.zero_grad()  # 清空网络中的梯度累计

            outputs = mlp(inputs)  # 前向传播计算预测结果
            loss = criterion(outputs, labels)  # 计算预测结果和实际标签的误差
            loss.backward()  # 将误差反向传播
            optimizer.step()  # 更新网络中的参数

            running_loss += loss.item()  # loss累计计算

            if epoch % opt.eval_freq == 0:
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        total_time += time.time() - start_time

        if epoch % opt.eval_freq == 0:
            x_axis.append(epoch)
            train_accuracy.append(correct / total)
            train_loss.append(running_loss / len(train_loader))
            acc, loss = evaluate(mlp, test_loader, criterion)
            test_accuracy.append(acc)
            test_loss.append(loss)
            print(f'Epoch {x_axis[-1]}\n'
                  f'Train Accuracy: {train_accuracy[-1]:.3f}\tTrain Loss: {train_loss[-1]:.3f}\t'
                  f'Test Accuracy: {test_accuracy[-1]:.3f}\tTest Loss: {test_loss[-1]:.3f}\t'
                  f'Avg Time Cost: {total_time / (epoch + 1):.3f}s')

    plt.plot(x_axis, train_accuracy, label='train accuracy')
    plt.plot(x_axis, train_loss, label='train loss')
    plt.plot(x_axis, test_accuracy, label='test accuracy')
    plt.plot(x_axis, test_loss, label='test loss')
    plt.legend()
    plt.show()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y, markers='o', s=25, edgecolor='k', legend=False)
    plt.show()


def main(opt):
    """
    Main function
    """
    data, label = None, None
    if opt.generator == 'moons':
        data, label = make_moons(n_samples=1000, shuffle=True, noise=0.05, random_state=opt.seed)
    elif opt.generator == 'circles':
        data, label = make_circles(n_samples=1000, shuffle=True, noise=0.05, random_state=opt.seed)
    x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=0.8, random_state=0)
    train(opt, x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--noise', type=float, default=0.05,
                        help='Noise of datasets')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--generator', choices=['moons', 'circles'], default='moons',
                        help='What datasets to generate')
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
