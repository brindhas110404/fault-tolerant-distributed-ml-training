import argparse
import json
import os
import time

import numpy as np

from worker.dataset import (
    load_data,
    shard_dataset,
    batches
)

from worker.model import (
    MLP,
    softmax_cross_entropy
)

from worker.optimizer import SGD


def train_single_worker(
    X,
    y,
    epochs,
    batch_size
):

    model = MLP(
        input_dim=X.shape[1],
        hidden_dim=128,
        num_classes=10,
        seed=7
    )

    optimizer = SGD(
        model.parameters(),
        learning_rate=0.05
    )

    history = []

    start_time = time.perf_counter()

    for epoch in range(epochs):

        losses = []

        for X_batch, y_batch in batches(
            X,
            y,
            batch_size
        ):

            logits = model.forward(
                X_batch
            )

            loss, gradient = (
                softmax_cross_entropy(
                    logits,
                    y_batch
                )
            )

            model.backward(
                gradient
            )

            optimizer.step(
                model.gradients()
            )

            losses.append(loss)

        elapsed = (
            time.perf_counter()
            - start_time
        )

        average_loss = np.mean(
            losses
        )

        print(
            f"Epoch {epoch + 1} | "
            f"Loss {average_loss:.4f} | "
            f"Time {elapsed:.2f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "loss": float(
                average_loss
            ),
            "time": elapsed
        })

    return history


def train_simulated_distributed(
    X,
    y,
    workers,
    epochs,
    batch_size
):

    shards = []

    for worker_id in range(
        workers
    ):

        shard = shard_dataset(
            X,
            y,
            worker_id,
            workers
        )

        shards.append(shard)

    models = [
        MLP(
            input_dim=X.shape[1],
            hidden_dim=128,
            num_classes=10,
            seed=7
        )
        for _ in range(workers)
    ]

    optimizers = [
        SGD(
            model.parameters(),
            learning_rate=0.05
        )
        for model in models
    ]

    start_time = time.perf_counter()

    history = []

    for epoch in range(epochs):

        worker_batches = [
            list(
                batches(
                    shard_X,
                    shard_y,
                    batch_size
                )
            )
            for shard_X, shard_y
            in shards
        ]

        number_of_steps = min(
            len(worker_batch)
            for worker_batch
            in worker_batches
        )

        epoch_losses = []

        for step in range(
            number_of_steps
        ):

            all_worker_gradients = []

            step_losses = []

            # ------------------------------------
            # LOCAL COMPUTATION
            # ------------------------------------

            for worker_id in range(
                workers
            ):

                model = models[
                    worker_id
                ]

                X_batch, y_batch = (
                    worker_batches[
                        worker_id
                    ][step]
                )

                logits = model.forward(
                    X_batch
                )

                loss, gradient = (
                    softmax_cross_entropy(
                        logits,
                        y_batch
                    )
                )

                model.backward(
                    gradient
                )

                gradients = [
                    gradient.copy()
                    for gradient
                    in model.gradients()
                ]

                all_worker_gradients.append(
                    gradients
                )

                step_losses.append(
                    loss
                )

            # ------------------------------------
            # ALLREDUCE
            # ------------------------------------

            averaged_gradients = []

            for parameter_index in range(
                len(
                    all_worker_gradients[0]
                )
            ):

                gradients = [
                    all_worker_gradients[
                        worker_id
                    ][parameter_index]
                    for worker_id in range(
                        workers
                    )
                ]

                averaged = np.mean(
                    gradients,
                    axis=0
                )

                averaged_gradients.append(
                    averaged
                )

            # ------------------------------------
            # SYNCHRONOUS UPDATE
            # ------------------------------------

            for optimizer in optimizers:

                optimizer.step(
                    averaged_gradients
                )

            epoch_losses.extend(
                step_losses
            )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        average_loss = np.mean(
            epoch_losses
        )

        print(
            f"Epoch {epoch + 1} | "
            f"Workers {workers} | "
            f"Loss {average_loss:.4f} | "
            f"Time {elapsed:.2f}s"
        )

        history.append({
            "epoch": epoch + 1,
            "loss": float(
                average_loss
            ),
            "time": elapsed
        })

    return history


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--workers",
        type=int,
        default=1
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--out",
        default="results/run.json"
    )

    args = parser.parse_args()

    X_train, X_test, y_train, y_test = (
        load_data()
    )

    if args.workers == 1:

        history = train_single_worker(
            X_train,
            y_train,
            args.epochs,
            args.batch_size
        )

    else:

        history = train_simulated_distributed(
            X_train,
            y_train,
            args.workers,
            args.epochs,
            args.batch_size
        )

    result = {
        "workers": args.workers,
        "epochs": args.epochs,
        "history": history
    }

    os.makedirs(
        "results",
        exist_ok=True
    )

    with open(
        args.out,
        "w"
    ) as file:

        json.dump(
            result,
            file,
            indent=2
        )


if __name__ == "__main__":

    main()