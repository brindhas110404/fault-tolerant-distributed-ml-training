"""Download checksum-verified official CIFAR-10 binary data without pickle/extraction."""
import argparse
import hashlib
from pathlib import Path
import tarfile
import urllib.request
import numpy as np

URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
MD5 = 'c32a1d4ab5d03f1284b67883e8d87530'


def prepare(data_dir):
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'cifar-10-binary.tar.gz'
    if not archive.exists():
        partial = archive.with_suffix('.download')
        urllib.request.urlretrieve(URL, partial)
        partial.replace(archive)
    with archive.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'md5').hexdigest()
    if checksum != MD5:
        raise ValueError('CIFAR-10 checksum mismatch; remove the archive and retry')
    target = root / 'cifar10'
    target.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for split, names in [('train', [f'data_batch_{i}.bin' for i in range(1, 6)]),
                             ('test', ['test_batch.bin'])]:
            records = []
            for name in names:
                with tar.extractfile('cifar-10-batches-bin/' + name) as stream:
                    values = np.frombuffer(stream.read(), dtype=np.uint8)
                if values.size != 10000 * 3073:
                    raise ValueError('unexpected CIFAR batch size')
                records.append(values.reshape(10000, 3073))
            records = np.concatenate(records)
            if records[:, 0].max() > 9:
                raise ValueError('invalid CIFAR labels')
            for key, value in [('X', records[:, 1:]), ('y', records[:, 0].astype(np.int64))]:
                destination = target / f'{key}_{split}.npy'
                temporary = destination.with_suffix('.tmp')
                with temporary.open('wb') as stream:
                    np.save(stream, value, allow_pickle=False)
                temporary.replace(destination)
    print(f'CIFAR-10 verified: 50,000 training and 10,000 test images in {target}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    prepare(parser.parse_args().data_dir)
