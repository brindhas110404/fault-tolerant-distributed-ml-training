import numpy as np

from worker.layers import (
    Dense,
    ReLU
)


def test_dense_forward_shape():

    layer = Dense(
        3,
        4
    )

    x = np.ones(
        (5, 3)
    )

    output = layer.forward(
        x
    )

    assert output.shape == (
        5,
        4
    )


def test_dense_backward_shape():

    layer = Dense(
        3,
        4
    )

    x = np.ones(
        (5, 3)
    )

    output = layer.forward(
        x
    )

    gradient = np.ones_like(
        output
    )

    dx = layer.backward(
        gradient
    )

    assert dx.shape == (
        5,
        3
    )


def test_relu():

    layer = ReLU()

    x = np.array([
        [-2.0, 3.0]
    ])

    output = layer.forward(
        x
    )

    assert np.allclose(
        output,
        [[0.0, 3.0]]
    )