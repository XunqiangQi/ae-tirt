"""Fit index calculations."""

import numpy as np


def aic(log_likelihood: float, n_params: int) -> float:
    return float(2 * n_params - 2 * log_likelihood)


def bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    return float(np.log(n_samples) * n_params - 2 * log_likelihood)
