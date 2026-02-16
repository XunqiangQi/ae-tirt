"""Evaluation metrics."""

import numpy as np
from scipy.stats import pearsonr


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_pred - y_true))


def correlation(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    if y_true.size < 2:
        return float("nan")
    cor, _ = pearsonr(y_true, y_pred)
    return float(cor)


def calculate_metrics(y_true, y_pred):
    return {"rmse": rmse(y_true, y_pred), "bias": bias(y_true, y_pred), "cor": correlation(y_true, y_pred)}


def evaluate_traits(estimated_traits, true_traits):
    """Trait recovery metrics with sign-flip correction."""
    true_traits = np.asarray(true_traits)
    estimated_traits = np.asarray(estimated_traits)
    n_persons, n_traits = true_traits.shape
    if estimated_traits.shape != (n_persons, n_traits):
        raise ValueError("estimated_traits and true_traits must have identical shape.")

    corrected_est = estimated_traits.copy()
    metrics = {"per_trait": {}, "overall": {}}

    for k in range(n_traits):
        true_k = true_traits[:, k]
        est_k = estimated_traits[:, k].copy()
        cor_k = correlation(true_k, est_k)
        if np.isfinite(cor_k) and cor_k < 0:
            est_k = -est_k
            corrected_est[:, k] = est_k
            cor_k = correlation(true_k, est_k)

        metrics["per_trait"][f"trait_{k + 1}"] = {
            "rmse": rmse(true_k, est_k),
            "bias": bias(true_k, est_k),
            "cor": cor_k,
            "rel": float(cor_k ** 2) if np.isfinite(cor_k) else float("nan"),
        }

    metrics["overall"] = {
        "rmse": rmse(true_traits, corrected_est),
        "bias": bias(true_traits, corrected_est),
        "cor": float(np.nanmean([m["cor"] for m in metrics["per_trait"].values()])),
        "rel": float(np.nanmean([m["rel"] for m in metrics["per_trait"].values()])),
    }
    return metrics
