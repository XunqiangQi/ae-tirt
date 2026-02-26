"""Reproduce paper-aligned real-data analysis results for AE-TIRT.

Loads pre-trained model weights and reports all evaluation metrics used in
the manuscript: model fit indices (-2LL, AIC, BIC), response reconstruction
metrics (ACC, AUC, F1, F2, Precision, Recall, Specificity, MCC), and
parameter estimates (theta, statement weights w, pair intercepts b).

Required inputs (default: data/real/)
--------------------------------------
    X_responses.csv     Binary response matrix (persons × pairs).
    item_trait_map.csv  Statement-to-trait mapping.
    pair_definitions.csv  Pair index definitions (item1, item2).
    weight_sign.csv     Statement loading sign constraints.

Usage
-----
    python scripts/run_real_data_analysis.py --weights data/real/pretrained_model.pth
    python scripts/run_real_data_analysis.py --weights data/real/pretrained_model.pth \\
        --data-dir data/real --device cpu
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score, precision_score, recall_score, matthews_corrcoef

from ae_tirt import AE_TIRT

SEED = 42


def _set_seed():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _read_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Load real data: responses, item_trait_map, pair_definitions, weight_sign."""
    resp_df = pd.read_csv(data_dir / "X_responses.csv")
    if "userid" in resp_df.columns:
        resp_df = resp_df.drop(columns=["userid"])
    responses = resp_df.values

    itm = pd.read_csv(data_dir / "item_trait_map.csv")
    if "trait_id" in itm.columns:
        itm = itm.sort_values("statement_id")
        item_trait_map = itm["trait_id"].to_numpy()
    else:
        item_trait_map = itm["item_trait"].to_numpy()

    pair_df = pd.read_csv(data_dir / "pair_definitions.csv")
    if "statement_j" in pair_df.columns:
        pair_df = pair_df.rename(columns={"statement_j": "item1", "statement_k": "item2"})
    pair_definitions = pair_df[["item1", "item2"]].values

    sign_df = pd.read_csv(data_dir / "weight_sign.csv")
    if "statement_id" in sign_df.columns:
        sign_df = sign_df.sort_values("statement_id")
    weight_sign = sign_df["weight_sign"].to_numpy()

    return responses, item_trait_map, pair_definitions, weight_sign


def _compute_fit_indices(model: AE_TIRT, x: torch.Tensor) -> dict:
    """Log-likelihood, -2LL, AIC, BIC."""
    model.eval()
    with torch.no_grad():
        p, z = model(x)
        log_lik = model.compute_log_likelihood(x, z).item()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_obs = int(x.shape[0] * x.shape[1])
    neg2ll = -2.0 * log_lik
    aic = neg2ll + 2 * n_params
    bic = neg2ll + n_params * np.log(n_obs)
    return {
        "log_likelihood": log_lik,
        "neg2ll": neg2ll,
        "AIC": aic,
        "BIC": bic,
        "n_params": n_params,
        "n_obs": n_obs,
    }


def _compute_reconstruction_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> dict:
    """ACC, AUC, F1, F2, Precision, Recall, Specificity, MCC."""
    y_flat = y_true.ravel().astype(int)
    p_flat = np.clip(p_pred.ravel(), 1e-8, 1 - 1e-8)
    pred = (p_flat >= 0.5).astype(int)

    acc = float(np.mean(pred == y_flat))
    auc = float(roc_auc_score(y_flat, p_flat)) if len(np.unique(y_flat)) >= 2 else float("nan")
    f1 = float(f1_score(y_flat, pred, zero_division=0))
    f2 = float(fbeta_score(y_flat, pred, beta=2.0, zero_division=0))
    precision = float(precision_score(y_flat, pred, zero_division=0))
    recall = float(recall_score(y_flat, pred, zero_division=0))

    tn = int(np.sum((pred == 0) & (y_flat == 0)))
    fp = int(np.sum((pred == 1) & (y_flat == 0)))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    mcc = float(matthews_corrcoef(y_flat, pred))

    return {
        "ACC": acc, "AUC": auc, "F1": f1, "F2": f2,
        "Precision": precision, "Recall": recall,
        "Specificity": specificity, "MCC": mcc,
    }


def _print_results(fit: dict, recon: dict, theta: np.ndarray, w: np.ndarray, b: np.ndarray):
    """Print results in publication format."""
    print("\n" + "=" * 70)
    print("AE-TIRT Real-Data Analysis Results (Pre-trained Model)")
    print("=" * 70)

    print("\n### Model Fit Indices ###")
    print(f"  -2LL:           {fit['neg2ll']:>12.4f}")
    print(f"  AIC:            {fit['AIC']:>12.4f}")
    print(f"  BIC:            {fit['BIC']:>12.4f}")
    print(f"  Log-likelihood: {fit['log_likelihood']:>12.4f}")
    print(f"  N parameters:   {fit['n_params']:>12d}")
    print(f"  N observations: {fit['n_obs']:>12d}")

    print("\n### Reconstruction Metrics ###")
    print(f"  ACC:         {recon['ACC']:>8.4f}")
    print(f"  AUC:         {recon['AUC']:>8.4f}")
    print(f"  F1:          {recon['F1']:>8.4f}")
    print(f"  F2:          {recon['F2']:>8.4f}")
    print(f"  Precision:   {recon['Precision']:>8.4f}")
    print(f"  Recall:      {recon['Recall']:>8.4f}")
    print(f"  Specificity: {recon['Specificity']:>8.4f}")
    print(f"  MCC:         {recon['MCC']:>8.4f}")

    print("\n### Parameter Estimates ###")
    print("\nTrait estimates (theta, first 5 persons):")
    K = theta.shape[1]
    theta_df = pd.DataFrame(theta[:5], columns=[f"trait_{k+1}" for k in range(K)])
    print(theta_df.round(4).to_string(index=True))

    print(f"\nStatement weights (w): N={len(w)}, min={w.min():.4f}, max={w.max():.4f}, mean={w.mean():.4f}")
    if len(w) <= 10:
        print(f"  Values: {np.round(w, 4).tolist()}")
    else:
        print(f"  First 5: {np.round(w[:5], 4).tolist()}")
        print(f"  Last 5:  {np.round(w[-5:], 4).tolist()}")

    print(f"\nPair intercepts (b): N={len(b)}, min={b.min():.4f}, max={b.max():.4f}, mean={b.mean():.4f}")
    if len(b) <= 10:
        print(f"  Values: {np.round(b, 4).tolist()}")
    else:
        print(f"  First 5: {np.round(b[:5], 4).tolist()}")
        print(f"  Last 5:  {np.round(b[-5:], 4).tolist()}")

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce paper-aligned AE-TIRT real-data analysis results."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="data/real/pretrained_model.pth",
        help="Path to pre-trained model weights (.pth).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/real",
        help="Directory containing the real-data CSV files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda or cpu).",
    )
    args = parser.parse_args()

    _set_seed()

    data_dir = Path(args.data_dir)
    weights_path = Path(args.weights)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    responses, item_trait_map, pair_definitions, weight_sign = _read_data(data_dir)
    N, L = responses.shape
    K = int(item_trait_map.max())

    print(f"Data: N={N} persons, L={L} pairs, K={K} traits, D={len(item_trait_map)} statements")
    print(f"Weights: {weights_path}")

    model = AE_TIRT(
        input_dim=L,
        latent_dim=K,
        item_trait_map=torch.tensor(item_trait_map, dtype=torch.long),
        pair_definitions=torch.tensor(pair_definitions, dtype=torch.long),
        weight_sign=torch.tensor(weight_sign, dtype=torch.float32),
        weight_constraint="standardized",
        link_function="probit",
    )

    device = torch.device(args.device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    x = torch.tensor(responses, dtype=torch.float32, device=device)
    with torch.no_grad():
        p, z = model(x)

    fit = _compute_fit_indices(model, x)
    recon = _compute_reconstruction_metrics(responses, p.cpu().numpy())
    _print_results(fit, recon, z.cpu().numpy(), model.weights.detach().cpu().numpy(), model.biases.detach().cpu().numpy())


if __name__ == "__main__":
    main()
