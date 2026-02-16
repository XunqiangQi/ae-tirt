"""Validation helpers."""

import numpy as np
import torch

from .metrics import calculate_metrics, evaluate_traits


def validate_equal_length(a, b, name_a="a", name_b="b"):
    if len(a) != len(b):
        raise ValueError(f"{name_a} and {name_b} must have same length")


def evaluate_model(model, test_data, true_traits, device="cpu"):
    """Evaluate trait recovery (latent estimates vs true traits)."""
    model.eval()
    test_data_tensor = torch.FloatTensor(test_data).to(device)
    with torch.no_grad():
        _, theta = model(test_data_tensor)
        theta_np = theta.cpu().numpy()
    return {"traits": evaluate_traits(theta_np, true_traits)}


def evaluate_item_parameters(model, true_params, pair_definitions=None, item_trait_map=None):
    """
    Evaluate recovery of TIRT item parameters (w, b).

    pair_definitions and item_trait_map are accepted for API compatibility.
    """
    del pair_definitions, item_trait_map
    estimated_params = model.get_trained_parameters()
    results = {"overall": {}, "per_parameter": {}}

    w_true = np.asarray(true_params["w"])
    b_true = np.asarray(true_params["b"])
    w_est = np.array([estimated_params["statement_weights"][i + 1] for i in range(len(w_true))])
    b_est = np.array([estimated_params["pair_intercepts"][l + 1] for l in range(len(b_true))])

    results["overall"] = {
        "w": calculate_metrics(w_true, w_est),
        "b": calculate_metrics(b_true, b_est),
    }

    for param in ("w", "b"):
        results["per_parameter"][param] = {"overall": results["overall"][param]}
    return results
