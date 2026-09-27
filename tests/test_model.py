import numpy as np

from worker.model import (
    MLP,
    softmax_cross_entropy
)


def test_model():

    model = MLP(
        input_dim=4,
        hidden_dim=8,
        num_classes=3
    )

    X = np.random.randn(
        5,
        4
    )

    y = np.array([
        0,
        1,
        2,
        1,
        0
    ])

    logits = model.forward(
        X
    )

    loss, gradient = (
        softmax_cross_entropy(
            logits,
            y
        )
    )

    assert np.isfinite(
        loss
    )

    model.backward(
        gradient
    )

    for g in model.gradients():

        assert np.all(
            np.isfinite(g)
        )