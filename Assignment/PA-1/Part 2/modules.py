import numpy as np


class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0.
        3) Initialize gradients with zeros.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.params = {
            'weight': np.random.normal(0, 1, size=(out_features, in_features)),
            'bias': np.zeros(shape=out_features)
        }
        self.grads = {
            'weight': np.zeros(shape=(out_features, in_features)),
            'bias': np.zeros(shape=out_features)
        }

    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object and use them in
        the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """
        self.params['x'] = np.squeeze(x)
        out = self.params['weight'] @ self.params['x'] + self.params['bias']
        return out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        self.grads['weight'] += np.expand_dims(dout, axis=1) @ np.expand_dims(self.params['x'].T, axis=0)
        self.grads['bias'] += dout
        return self.params['weight'].T @ dout

    def step(self, n, lr):
        self.params['weight'] -= lr * self.grads['weight'] / n
        self.params['bias'] -= lr * self.grads['bias'] / n

    def zero_grad(self):
        self.grads['weight'] = np.zeros(shape=(self.out_features, self.in_features))
        self.grads['bias'] = np.zeros(shape=self.out_features)


class ReLU(object):
    def __init__(self):
        self.params = dict()

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        self.params['x'] = x
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        return np.where(self.params['x'] > 0, dout, 0)


class SoftMax(object):
    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        """
        b = np.max(x)
        y = np.exp(x - b)
        self.out = y / y.sum()
        return self.out

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        return self.out * (dout - np.sum(dout * self.out, axis=0))


class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        return -np.sum(y * np.log(x + 1e-5))

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        return -y / (x + 1e-5)
