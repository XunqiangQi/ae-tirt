"""Data preprocessing utilities."""

import numpy as np


def to_float32(array):
    return np.asarray(array, dtype=np.float32)
