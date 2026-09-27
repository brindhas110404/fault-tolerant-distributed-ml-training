import io

import numpy as np


def pack_array(array):

    buffer = io.BytesIO()

    np.save(
        buffer,
        np.asarray(array),
        allow_pickle=False
    )

    return buffer.getvalue()


def unpack_array(payload):

    buffer = io.BytesIO(
        payload
    )

    return np.load(
        buffer,
        allow_pickle=False
    )