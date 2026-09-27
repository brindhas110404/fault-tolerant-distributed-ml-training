import numpy as np


class Dense:

    def __init__(self, in_features, out_features, rng=None):

        rng = rng or np.random.default_rng()

        self.W = (
            rng.normal(
                0,
                np.sqrt(2.0 / in_features),
                (in_features, out_features)
            )
        )

        self.b = np.zeros((1, out_features))

        self.x = None

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):

        self.x = x

        return x @ self.W + self.b

    def backward(self, grad):

        self.dW = self.x.T @ grad

        self.db = np.sum(
            grad,
            axis=0,
            keepdims=True
        )

        return grad @ self.W.T


class ReLU:

    def __init__(self):

        self.mask = None

    def forward(self, x):

        self.mask = x > 0

        return np.maximum(x, 0)

    def backward(self, grad):

        return grad * self.mask


class Flatten:

    def __init__(self):

        self.original_shape = None

    def forward(self, x):

        self.original_shape = x.shape

        return x.reshape(
            x.shape[0],
            -1
        )

    def backward(self, grad):

        return grad.reshape(
            self.original_shape
        )