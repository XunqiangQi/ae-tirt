"""Data transforms."""

import numpy as np


def binarize(x, threshold=0.5):
    arr = np.asarray(x)
    return (arr >= threshold).astype(int)
