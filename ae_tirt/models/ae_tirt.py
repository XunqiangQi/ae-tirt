"""AE-TIRT model core."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from ae_tirt.utils.paths import ensure_dir, get_safe_path
from .base import BaseIRTModel
from .components import build_encoder


class AE_TIRT(BaseIRTModel):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        item_trait_map: torch.Tensor,
        pair_definitions: torch.Tensor,
        weight_sign: torch.Tensor,
        weight_constraint: str = "free",
        link_function: str = "probit",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.D = item_trait_map.shape[0]
        self.L = input_dim
        self.K = latent_dim

        self.encoder = build_encoder(input_dim, latent_dim)

        self.item_trait_map = self._process_indices(item_trait_map, latent_dim, "trait")
        self.pair_definitions = self._process_indices(pair_definitions, item_trait_map.shape[0], "statement")

        weight_constraint = str(weight_constraint).lower().strip()
        if weight_constraint not in {"standardized", "free"}:
            raise ValueError("weight_constraint must be 'standardized' or 'free'")
        self.weight_constraint = weight_constraint
        self.register_buffer("weight_sign", weight_sign)
        self.raw_magnitude = nn.Parameter(torch.rand(item_trait_map.shape[0]))
        self.biases = nn.Parameter(torch.zeros(input_dim))

        link_function = str(link_function).lower().strip()
        if link_function not in {"probit", "logit"}:
            raise ValueError("link_function must be 'probit' or 'logit'")
        self.link_function = link_function

        self._validate_shapes()
        self._validate_cross_trait_pairs()

    @property
    def weights(self):
        if self.weight_constraint == "standardized":
            magnitude = torch.sigmoid(self.raw_magnitude)
        else:
            magnitude = torch.abs(self.raw_magnitude)
        return magnitude * self.weight_sign

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = 3 * torch.tanh(h)
        recon_x = self._decode(z)
        return recon_x, z

    def loss_function(
        self, x: torch.Tensor, recon_x: torch.Tensor, z: torch.Tensor, penalty_weight: float = 0.1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
        z_penalty = torch.mean(z ** 2)
        total_loss = recon_loss + penalty_weight * z_penalty
        return total_loss, recon_loss, z_penalty

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        j = self.pair_definitions[:, 0]
        k = self.pair_definitions[:, 1]
        a = self.item_trait_map[j]
        b = self.item_trait_map[k]
        theta_a = z[:, a]
        theta_b = z[:, b]
        w_j = self.weights[j]
        w_k = self.weights[k]
        linear = w_j * theta_a - w_k * theta_b + self.biases
        if self.link_function == "logit":
            return torch.sigmoid(linear)
        return Normal(0, 1).cdf(linear)

    def print_trait_estimates(self, x, num_examples=5, decimal_places=4):
        with torch.no_grad():
            _, z = self.forward(x)
        df = pd.DataFrame(z.cpu().numpy(), columns=[f"trait_{k+1}" for k in range(self.K)])
        print(df.head(num_examples).round(decimal_places).to_markdown(tablefmt="grid"))

    def _process_indices(self, tensor, max_val, name):
        if tensor.min() == 1:
            tensor = tensor - 1
        if tensor.min() < 0 or tensor.max() >= max_val:
            raise ValueError(f"{name} index out of range")
        return tensor.long()

    def _validate_shapes(self):
        if self.item_trait_map.shape != (self.D,):
            raise ValueError("item_trait_map shape invalid")
        if self.pair_definitions.shape != (self.L, 2):
            raise ValueError("pair_definitions shape invalid")

    def _validate_cross_trait_pairs(self):
        error_pairs = []
        for l in range(self.L):
            j, k = self.pair_definitions[l]
            if self.item_trait_map[j] == self.item_trait_map[k]:
                error_pairs.append((l + 1, j + 1, k + 1))
        if error_pairs:
            raise ValueError(f"Same-trait pairs not allowed: {error_pairs[:3]}")

    def get_trained_parameters(self):
        return {
            "statement_weights": {i + 1: self.weights[i].item() for i in range(self.D)},
            "pair_intercepts": {i + 1: self.biases[i].item() for i in range(self.L)},
        }

    def compute_log_likelihood(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute Bernoulli log-likelihood under decoder probabilities."""
        probs = self._decode(z)
        eps = 1e-8
        probs = torch.clamp(probs, eps, 1 - eps)
        return torch.sum(x * torch.log(probs) + (1 - x) * torch.log(1 - probs))

    def compute_standard_errors(
        self,
        x: torch.Tensor,
        method: str = "observed",
        use_hessian_diag: bool = False,
        regularization: float = 1e-4,
        verbose: bool = False,
    ) -> dict[str, np.ndarray | str | bool]:
        """
        Compute post-hoc standard errors for all parameters.

        Notes
        -----
        - This is an optional post-estimation step and is never run by default.
        - The method is based on the observed information from the decoder likelihood.
        """
        method = str(method).lower().strip()
        if method not in {"observed", "expected"}:
            raise ValueError("method must be 'observed' or 'expected'")
        if regularization <= 0:
            raise ValueError("regularization must be positive")

        self.eval()
        x_eval = x.detach()
        device = x_eval.device
        n_persons, n_pairs = x_eval.shape
        n_traits = self.latent_dim
        n_items = self.D

        with torch.no_grad():
            _, z = self.forward(x_eval)

        z_param = z.detach().clone().requires_grad_(True)
        raw_magnitude_param = self.raw_magnitude.detach().clone().requires_grad_(True)
        biases_param = self.biases.detach().clone().requires_grad_(True)
        weight_sign_param = self.weight_sign.to(device)

        if self.weight_constraint == "standardized":
            magnitude = torch.sigmoid(raw_magnitude_param)
        else:
            magnitude = torch.abs(raw_magnitude_param)
        weights_param = magnitude * weight_sign_param

        j = self.pair_definitions[:, 0]
        k = self.pair_definitions[:, 1]
        a = self.item_trait_map[j]
        b = self.item_trait_map[k]

        theta_a = z_param[:, a]
        theta_b = z_param[:, b]
        w_j = weights_param[j]
        w_k = weights_param[k]
        linear = w_j * theta_a - w_k * theta_b + biases_param

        if self.link_function == "logit":
            probs = torch.sigmoid(linear)
        else:
            probs = Normal(0, 1).cdf(linear)

        eps = 1e-8
        probs = torch.clamp(probs, eps, 1 - eps)
        log_lik = torch.sum(x_eval * torch.log(probs) + (1 - x_eval) * torch.log(1 - probs))

        n_theta = n_persons * n_traits
        n_w = n_items
        n_b = n_pairs
        n_params = n_theta + n_w + n_b

        if verbose:
            print(f"Computing SEs for {n_params} parameters")
            print(f"  theta: {n_theta}, w: {n_w}, b: {n_b}")

        if use_hessian_diag:
            grads = torch.autograd.grad(
                log_lik,
                [z_param, raw_magnitude_param, biases_param],
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            grad_z = grads[0].reshape(-1) if grads[0] is not None else torch.zeros(n_theta, device=device)
            grad_w = grads[1] if grads[1] is not None else torch.zeros(n_w, device=device)
            grad_b = grads[2] if grads[2] is not None else torch.zeros(n_b, device=device)

            hess_diag_z = torch.zeros(n_theta, device=device)
            hess_diag_w = torch.zeros(n_w, device=device)
            hess_diag_b = torch.zeros(n_b, device=device)

            for i in range(n_theta):
                if grad_z[i].requires_grad:
                    grad2 = torch.autograd.grad(grad_z[i], z_param, retain_graph=True, allow_unused=True)[0]
                    if grad2 is not None:
                        hess_diag_z[i] = grad2.reshape(-1)[i]

            for i in range(n_w):
                if grad_w[i].requires_grad:
                    grad2 = torch.autograd.grad(grad_w[i], raw_magnitude_param, retain_graph=True, allow_unused=True)[0]
                    if grad2 is not None:
                        hess_diag_w[i] = grad2[i]

            for i in range(n_b):
                if grad_b[i].requires_grad:
                    grad2 = torch.autograd.grad(grad_b[i], biases_param, retain_graph=True, allow_unused=True)[0]
                    if grad2 is not None:
                        hess_diag_b[i] = grad2[i]

            info_diag_z = -hess_diag_z + regularization
            info_diag_w = -hess_diag_w + regularization
            info_diag_b = -hess_diag_b + regularization

            se_z = torch.sqrt(1.0 / torch.abs(info_diag_z)).reshape(n_persons, n_traits)
            se_w = torch.sqrt(1.0 / torch.abs(info_diag_w))
            se_b = torch.sqrt(1.0 / torch.abs(info_diag_b))
        else:
            if verbose:
                print("Computing full Hessian; this may be slow on large N")

            hessian = torch.zeros(n_params, n_params, device=device)
            grads = torch.autograd.grad(
                log_lik,
                [z_param, raw_magnitude_param, biases_param],
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )

            grad_z_full = grads[0] if grads[0] is not None else torch.zeros(n_persons, n_traits, device=device)
            grad_w_full = grads[1] if grads[1] is not None else torch.zeros(n_w, device=device)
            grad_b_full = grads[2] if grads[2] is not None else torch.zeros(n_b, device=device)
            grad_flat = torch.cat([grad_z_full.reshape(-1), grad_w_full, grad_b_full])

            for i in range(n_params):
                if verbose and i % 100 == 0:
                    print(f"  Hessian row {i}/{n_params}")
                if grad_flat[i].requires_grad:
                    grad2 = torch.autograd.grad(
                        grad_flat[i],
                        [z_param, raw_magnitude_param, biases_param],
                        retain_graph=True,
                        allow_unused=True,
                    )
                    grad2_flat = torch.cat(
                        [
                            grad2[0].reshape(-1) if grad2[0] is not None else torch.zeros(n_theta, device=device),
                            grad2[1] if grad2[1] is not None else torch.zeros(n_w, device=device),
                            grad2[2] if grad2[2] is not None else torch.zeros(n_b, device=device),
                        ]
                    )
                    hessian[i, :] = grad2_flat

            info_matrix = -hessian
            info_matrix += torch.eye(n_params, device=device) * regularization

            try:
                cov_matrix = torch.inverse(info_matrix)
                se_flat = torch.sqrt(torch.abs(torch.diag(cov_matrix)))
            except RuntimeError:
                info_diag = torch.diag(info_matrix)
                se_flat = torch.sqrt(1.0 / torch.abs(info_diag))

            se_z = se_flat[:n_theta].reshape(n_persons, n_traits)
            se_w = se_flat[n_theta : n_theta + n_w]
            se_b = se_flat[n_theta + n_w :]

        if self.weight_constraint == "standardized":
            sigmoid_vals = torch.sigmoid(self.raw_magnitude)
            derivative = sigmoid_vals * (1 - sigmoid_vals)
            se_w_actual = se_w * torch.abs(derivative)
        else:
            se_w_actual = se_w

        result: dict[str, np.ndarray | str | bool] = {
            "theta_est": z.detach().cpu().numpy(),
            "theta_se": se_z.detach().cpu().numpy(),
            "w_se": se_w_actual.detach().cpu().numpy(),
            "b_se": se_b.detach().cpu().numpy(),
            "method": method,
            "use_diagonal": use_hessian_diag,
        }

        return result

    @staticmethod
    def _wald_stats(estimates: np.ndarray, ses: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Wald z, two-sided p, and normal-approximation CI bounds."""
        eps = 1e-12
        z_vals = estimates / np.clip(ses, eps, None)
        # two-sided p-value: p = 2 * (1 - Phi(|z|))
        z_tensor = torch.as_tensor(np.abs(z_vals), dtype=torch.float32)
        p_vals = (2.0 * (1.0 - Normal(0, 1).cdf(z_tensor))).cpu().numpy()
        p_vals = np.clip(p_vals, 0.0, 1.0)
        z_alpha = Normal(0, 1).icdf(torch.tensor(1 - alpha / 2)).item()
        ci_low = estimates - z_alpha * ses
        ci_high = estimates + z_alpha * ses
        return z_vals, p_vals, ci_low, ci_high

    def build_standard_error_report(
        self,
        se_results: dict,
        alpha: float = 0.05,
        include_theta: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Build academic-style SE report tables.

        Returns DataFrames with columns:
        estimate, SE, z, p, CI lower, CI upper.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

        reports: dict[str, pd.DataFrame] = {}
        ci_label_low = f"CI{int((1 - alpha) * 100)}_low"
        ci_label_high = f"CI{int((1 - alpha) * 100)}_high"

        w_est = self.weights.detach().cpu().numpy()
        w_se = np.asarray(se_results["w_se"])
        w_z, w_p, w_lo, w_hi = self._wald_stats(w_est, w_se, alpha=alpha)
        reports["w"] = pd.DataFrame(
            {
                "parameter": [f"w_{i+1}" for i in range(len(w_est))],
                "estimate": w_est,
                "SE": w_se,
                "z": w_z,
                "p": w_p,
                ci_label_low: w_lo,
                ci_label_high: w_hi,
            }
        )

        b_est = self.biases.detach().cpu().numpy()
        b_se = np.asarray(se_results["b_se"])
        b_z, b_p, b_lo, b_hi = self._wald_stats(b_est, b_se, alpha=alpha)
        reports["b"] = pd.DataFrame(
            {
                "parameter": [f"b_{i+1}" for i in range(len(b_est))],
                "estimate": b_est,
                "SE": b_se,
                "z": b_z,
                "p": b_p,
                ci_label_low: b_lo,
                ci_label_high: b_hi,
            }
        )

        if include_theta:
            if "theta_est" not in se_results:
                raise ValueError("se_results must include 'theta_est' to report theta parameters")
            theta_est = np.asarray(se_results["theta_est"])
            theta_se = np.asarray(se_results["theta_se"])
            if theta_est.shape != theta_se.shape:
                raise ValueError("theta_est and theta_se must have the same shape")
            t_z, t_p, t_lo, t_hi = self._wald_stats(theta_est, theta_se, alpha=alpha)
            theta_rows = []
            for person_idx in range(theta_est.shape[0]):
                for trait_idx in range(theta_est.shape[1]):
                    theta_rows.append(
                        {
                            "person": person_idx + 1,
                            "trait": trait_idx + 1,
                            "parameter": f"theta_{person_idx+1}_{trait_idx+1}",
                            "estimate": theta_est[person_idx, trait_idx],
                            "SE": theta_se[person_idx, trait_idx],
                            "z": t_z[person_idx, trait_idx],
                            "p": t_p[person_idx, trait_idx],
                            ci_label_low: t_lo[person_idx, trait_idx],
                            ci_label_high: t_hi[person_idx, trait_idx],
                        }
                    )
            reports["theta"] = pd.DataFrame(theta_rows)

        return reports

    def print_standard_errors_summary(
        self,
        se_results: dict,
        num_examples: int = 5,
        decimal_places: int = 4,
        alpha: float = 0.05,
    ):
        """Print a publication-style SE summary (estimate, SE, z, p, CI)."""
        print("\n=== Standard Errors Summary ===")
        print(f"Method: {se_results['method']}, Diagonal approximation: {se_results['use_diagonal']}")
        reports = self.build_standard_error_report(se_results=se_results, alpha=alpha, include_theta=True)

        print("\n--- Theta parameters (first rows) ---")
        print(reports["theta"].head(num_examples).round(decimal_places).to_markdown(index=False, tablefmt="grid"))

        print("\n--- Loading parameters (first rows) ---")
        print(reports["w"].head(num_examples).round(decimal_places).to_markdown(index=False, tablefmt="grid"))

        print("\n--- Intercept parameters (first rows) ---")
        print(reports["b"].head(num_examples).round(decimal_places).to_markdown(index=False, tablefmt="grid"))

    def save_standard_errors(
        self,
        se_results: dict,
        output_dir: str = "TIRT_eval",
        repeat: int | None = None,
        alpha: float = 0.05,
    ) -> dict[str, str]:
        """
        Save SE reports for theta, w, and b to CSV files.

        The export is inference-oriented and does not include true parameters.
        """
        if not isinstance(output_dir, str):
            output_dir = str(output_dir)
        if output_dir.startswith("/"):
            resolved_output_dir = output_dir
        else:
            resolved_output_dir = get_safe_path(output_dir)
        ensure_dir(resolved_output_dir)

        reports = self.build_standard_error_report(se_results=se_results, alpha=alpha, include_theta=True)
        ci_low_col = f"CI{int((1 - alpha) * 100)}_low"
        ci_high_col = f"CI{int((1 - alpha) * 100)}_high"

        theta_file = get_safe_path("theta_standard_errors.csv", base_dir=resolved_output_dir)
        loading_file = get_safe_path("loading_standard_errors.csv", base_dir=resolved_output_dir)
        intercept_file = get_safe_path("intercept_standard_errors.csv", base_dir=resolved_output_dir)
        summary_file = get_safe_path("standard_errors_summary.csv", base_dir=resolved_output_dir)

        theta_df = reports["theta"].copy()
        w_df = reports["w"].copy()
        b_df = reports["b"].copy()
        if repeat is not None:
            theta_df.insert(0, "repeat", repeat)
            w_df.insert(0, "repeat", repeat)
            b_df.insert(0, "repeat", repeat)

        theta_df.to_csv(theta_file, index=False)
        w_df.to_csv(loading_file, index=False)
        b_df.to_csv(intercept_file, index=False)

        summary_df = pd.DataFrame(
            {
                "parameter_group": ["theta", "loading", "intercept"],
                "mean_estimate": [theta_df["estimate"].mean(), w_df["estimate"].mean(), b_df["estimate"].mean()],
                "mean_SE": [theta_df["SE"].mean(), w_df["SE"].mean(), b_df["SE"].mean()],
                "median_SE": [theta_df["SE"].median(), w_df["SE"].median(), b_df["SE"].median()],
                "min_SE": [theta_df["SE"].min(), w_df["SE"].min(), b_df["SE"].min()],
                "max_SE": [theta_df["SE"].max(), w_df["SE"].max(), b_df["SE"].max()],
                "method": [se_results["method"], se_results["method"], se_results["method"]],
                "use_diagonal": [se_results["use_diagonal"], se_results["use_diagonal"], se_results["use_diagonal"]],
                "ci_low_col": [ci_low_col, ci_low_col, ci_low_col],
                "ci_high_col": [ci_high_col, ci_high_col, ci_high_col],
            }
        )
        if repeat is not None:
            summary_df.insert(0, "repeat", repeat)
        summary_df.to_csv(summary_file, index=False)

        return {
            "theta": theta_file,
            "w": loading_file,
            "b": intercept_file,
            "summary": summary_file,
        }
