import numpy as np


class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=1e-2):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.n = n_inputs
        self.epochs = int(max_epochs)
        self.lr = learning_rate
        self.w = np.zeros(shape=self.n)

    # noinspection PyShadowingBuiltins
    def forward(self, input):
        """
        Predict label from input 
        Args:
            input: array of dimension equal to n_inputs.
        """
        return np.sign(self.w @ input)

    def train(self, training_inputs, labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        for epoch in range(self.epochs):
            for i in range(len(labels)):
                predict = self.forward(training_inputs[i])
                if predict != labels[i]:
                    self.w += self.lr * (labels[i] * training_inputs[i])

    def predict(self, inputs):
        labels = []
        for point in inputs:
            labels.append(self.forward(point))
        return np.array(labels)

    def accuracy(self, test_data, test_labels):
        predict = self.predict(test_data)
        return np.sum(predict == test_labels) / len(test_labels)
