"""Model training loop."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .optimizers import build_optimizer


def train_model_core(model, train_loader, optimizer, num_epochs, early_stopping_patience, penalty_weight, device):
    """Core training loop for AE_TIRT."""
    history = {"total_loss": [], "recon_loss": [], "z_penalty": [], "epoch_times": []}
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        epoch_losses, epoch_recon_losses, epoch_z_penalties = [], [], []

        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, z = model(x)
            loss, recon_loss, z_penalty = model.loss_function(x, recon_x, z, penalty_weight)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_recon_losses.append(recon_loss.item())
            epoch_z_penalties.append(z_penalty.item())

        avg_loss = float(np.mean(epoch_losses))
        history["total_loss"].append(avg_loss)
        history["recon_loss"].append(float(np.mean(epoch_recon_losses)))
        history["z_penalty"].append(float(np.mean(epoch_z_penalties)))
        history["epoch_times"].append(time.time() - epoch_start_time)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, "
                f"Recon Loss: {np.mean(epoch_recon_losses):.4f}, "
                f"Z Penalty: {np.mean(epoch_z_penalties):.4f}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    return history


def train_model(
    model,
    train_data,
    optimizer_name="adam",
    batch_size=16,
    num_epochs=500,
    learning_rate=0.001,
    device="cpu",
    early_stopping_patience=20,
    penalty_weight=0.1,
    lr=None,
):
    """Train AE_TIRT and return history."""
    effective_lr = learning_rate if lr is None else lr
    model = model.to(device)
    train_data_tensor = torch.FloatTensor(train_data).to(device)
    train_dataset = TensorDataset(train_data_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = build_optimizer(model, name=optimizer_name, lr=effective_lr)
    return train_model_core(
        model,
        train_loader,
        optimizer,
        num_epochs,
        early_stopping_patience,
        penalty_weight,
        device,
    )
