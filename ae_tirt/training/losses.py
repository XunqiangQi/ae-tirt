"""Loss functions for training."""

import torch
import torch.nn as nn


def ae_tirt_loss(x: torch.Tensor, recon_x: torch.Tensor, z: torch.Tensor, penalty_weight: float = 0.1):
    recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    z_penalty = torch.mean(z ** 2)
    total_loss = recon_loss + penalty_weight * z_penalty
    return total_loss, recon_loss, z_penalty
