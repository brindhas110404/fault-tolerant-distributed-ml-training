import numpy as np
import pytest
from worker.dataset import load_dataset


def test_cifar_loader_preserves_official_split_and_uint8(tmp_path):
    root = tmp_path / 'cifar10'
    root.mkdir()
    for name, value in {'X_train': np.zeros((5, 3072), dtype=np.uint8),
                        'X_test': np.full((2, 3072), 255, dtype=np.uint8),
                        'y_train': np.arange(5), 'y_test': np.array([8, 9])}.items():
        np.save(root / f'{name}.npy', value)
    train, test, labels, test_labels = load_dataset('cifar10', tmp_path)
    assert isinstance(train, np.memmap)
    assert train.dtype == np.uint8 and train.shape == (5, 3072)
    assert test.shape == (2, 3072) and test.max() == 255
    assert list(test_labels) == [8, 9]


def test_missing_cifar_has_actionable_error(tmp_path):
    with pytest.raises(RuntimeError, match='prepare_cifar'):
        load_dataset('cifar10', tmp_path)
