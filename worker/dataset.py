import numpy as np

from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split


def load_data(
    test_size=0.2,
    seed=42
):

    dataset = load_digits()

    X = dataset.data.astype(
        np.float64
    )

    y = dataset.target.astype(
        np.int64
    )

    # Normalize pixel values
    X /= 16.0

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )


def shard_dataset(
    X,
    y,
    worker_id,
    world_size
):

    indices = np.arange(
        len(X)
    )

    # Same deterministic shuffle for every worker
    rng = np.random.default_rng(
        12345
    )

    rng.shuffle(indices)

    shards = np.array_split(
        indices,
        world_size
    )

    worker_indices = shards[
        worker_id
    ]

    return (
        X[worker_indices],
        y[worker_indices]
    )


def batches(
    X,
    y,
    batch_size
):

    for start in range(
        0,
        len(X),
        batch_size
    ):

        end = start + batch_size

        yield (
            X[start:end],
            y[start:end]
        )

def load_dataset(name='digits', data_dir='data'):
    """Return official train/test splits; CIFAR stays uint8 until minibatch use."""
    if name == 'digits':
        return load_data()
    if name != 'cifar10':
        raise ValueError(f'unknown dataset: {name}')
    from pathlib import Path
    root = Path(data_dir) / 'cifar10'
    try:
        return tuple(np.load(root / f'{key}.npy', mmap_mode='r', allow_pickle=False)
                     for key in ('X_train', 'X_test', 'y_train', 'y_test'))
    except FileNotFoundError as exc:
        raise RuntimeError('CIFAR-10 missing; run python -m worker.prepare_cifar --data-dir ' + str(data_dir)) from exc
