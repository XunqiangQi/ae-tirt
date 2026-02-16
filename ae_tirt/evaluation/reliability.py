"""Reliability estimators."""

import numpy as np


def empirical_reliability(theta_hat, theta_true):
    theta_hat = np.asarray(theta_hat)
    theta_true = np.asarray(theta_true)
    err_var = np.var(theta_hat - theta_true)
    true_var = np.var(theta_true)
    if true_var <= 0:
        return float("nan")
    return float(1 - (err_var / true_var))
