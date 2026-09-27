import numpy as np

from worker.layers import Dense, ReLU


def softmax_cross_entropy(logits, labels):

    # Numerical stability
    shifted = (
        logits
        - np.max(
            logits,
            axis=1,
            keepdims=True
        )
    )

    exp_values = np.exp(shifted)

    probabilities = (
        exp_values
        / np.sum(
            exp_values,
            axis=1,
            keepdims=True
        )
    )

    n = len(labels)

    loss = -np.mean(
        np.log(
            probabilities[
                np.arange(n),
                labels
            ] + 1e-12
        )
    )

    gradient = probabilities.copy()

    gradient[
        np.arange(n),
        labels
    ] -= 1

    gradient /= n

    return float(loss), gradient


class MLP:

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_classes,
        seed=7
    ):

        rng = np.random.default_rng(seed)

        self.fc1 = Dense(
            input_dim,
            hidden_dim,
            rng
        )

        self.relu = ReLU()

        self.fc2 = Dense(
            hidden_dim,
            num_classes,
            rng
        )

    def forward(self, x):

        x = self.fc1.forward(x)

        x = self.relu.forward(x)

        x = self.fc2.forward(x)

        return x

    def backward(self, gradient):

        gradient = self.fc2.backward(
            gradient
        )

        gradient = self.relu.backward(
            gradient
        )

        gradient = self.fc1.backward(
            gradient
        )

        return gradient

    def parameters(self):

        return [
            self.fc1.W,
            self.fc1.b,
            self.fc2.W,
            self.fc2.b
        ]

    def gradients(self):

        return [
            self.fc1.dW,
            self.fc1.db,
            self.fc2.dW,
            self.fc2.db
        ]