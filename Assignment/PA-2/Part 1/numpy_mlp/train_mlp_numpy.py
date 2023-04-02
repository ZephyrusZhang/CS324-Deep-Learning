from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mlp_numpy import MLP
from modules import CrossEntropy
from sklearn.datasets import make_moons, make_circles

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10
METHOD_DEFAULT = 'BGD'

FLAGS = None


def make_data(noise: float, n=1000, train_size=0.8, shuffle=True):
    train_data_size = int(n * train_size)
    x, y = None, None
    if FLAGS.generator == 'moons':
        x, y = make_moons(n_samples=n, shuffle=shuffle, noise=noise, random_state=FLAGS.seed)
    elif FLAGS.generator == 'circles':
        x, y = make_circles(n_samples=n, shuffle=shuffle, noise=noise, random_state=FLAGS.seed)
    # x, y = make_moons(n_samples=n, shuffle=shuffle, noise=noise)
    y = np.column_stack((y, 1 - y))
    x_train, x_test = x[:train_data_size], x[train_data_size:]
    y_train, y_test = y[:train_data_size], y[train_data_size:]
    return x_train, x_test, y_train, y_test


def accuracy(predictions: np.ndarray, targets: np.ndarray):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    cnt = 0
    for i in range(len(predictions)):
        if (np.around(predictions[i]) == targets[i]).all():
            cnt += 1
    return cnt / len(predictions)


def accuracy_loss(mlp: MLP, data: np.ndarray, label: np.ndarray, criterion: CrossEntropy):
    loss = 0
    predict = []
    for x, y in zip(data, label):
        y_hat = mlp.forward(x)
        loss += criterion.forward(y_hat, y)
        predict.append(y_hat)
    return accuracy(np.array(predict), label), loss / len(data)


def train(opt,
          x_train=None,
          x_test=None,
          y_train=None,
          y_test=None):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    print(opt)
    # YOUR TRAINING CODE GOES HERE
    x_axis = []
    train_accuracy, train_loss, test_accuracy, test_loss = [], [], [], []

    n_hidden = list(map(lambda ele: int(ele), opt.dnn_hidden_units.split(',')))
    lr = opt.learning_rate
    epochs = opt.max_steps
    eval_freq = opt.eval_freq
    method = opt.method

    batch_size = x_train.shape[0]
    mlp = MLP(2, n_hidden, 2, batch_size, lr)
    criterion = CrossEntropy()
    for epoch in range(epochs):
        mlp.zero_grad()
        for i in range(batch_size):
            i = i if method == 'BGD' else np.random.randint(0, batch_size)
            x, y = x_train[i], y_train[i]

            y_hat = mlp.forward(x)
            dx = criterion.backward(y_hat, y)
            mlp.backward(dx)
            if method == 'SGD':
                mlp.step()
        if method == 'BGD':
            mlp.step()

        if epoch % eval_freq == 0:
            x_axis.append(epoch)

            acc, loss = accuracy_loss(mlp, x_train, y_train, criterion)
            train_accuracy.append(acc)
            train_loss.append(loss)
            acc, loss = accuracy_loss(mlp, x_test, y_test, criterion)
            test_accuracy.append(acc)
            test_loss.append(loss)

            print(f'Epoch {x_axis[-1]} \t\t'
                  f'Train Accuracy: {train_accuracy[-1]:.3f} \t Train Loss: {train_loss[-1]:.3f}\t'
                  f'Test Accuracy: {test_accuracy[-1]:.3f} \t Test Loss: {test_loss[-1]:.3f}')

    plt.plot(x_axis, train_accuracy, label='train accuracy')
    plt.plot(x_axis, train_loss, label='train loss')
    plt.plot(x_axis, test_accuracy, label='test accuracy')
    plt.plot(x_axis, test_loss, label='test loss')
    plt.legend()
    plt.show()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))
    sns.scatterplot(x=x[:, 0], y=x[:, 1], hue=y[:, 0], markers='o', s=25, edgecolor='k', legend=False)
    plt.show()


def main(opt):
    """
    Main function
    """
    np.random.seed(opt.seed)
    x_train, x_test, y_train, y_test = make_data(opt.noise)
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
    parser.add_argument('--method', choices=['BGD', 'SGD'], default='BGD',
                        help='Method of gradient descent')
    parser.add_argument('--noise', type=float, default=0.05,
                        help='Noise of datasets')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--generator', choices=['moons', 'circles'], default='moons',
                        help='What datasets to generate')
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
