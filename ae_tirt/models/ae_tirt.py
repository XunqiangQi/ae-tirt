"""AE-TIRT model core."""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

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
