"""Reusable model components."""

import torch.nn as nn


def build_encoder(input_dim: int, latent_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.Sigmoid(),
        nn.Linear(input_dim, input_dim),
        nn.Sigmoid(),
        nn.Linear(input_dim, input_dim),
        nn.Sigmoid(),
        nn.Linear(input_dim, latent_dim),
    )
