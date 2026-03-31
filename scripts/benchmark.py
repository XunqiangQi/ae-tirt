"""Lightweight runtime benchmark for AE-TIRT training loop."""

from __future__ import annotations

import time

import torch

from ae_tirt import AE_TIRT, Sim_data_TIRT, train_model


def main():
    sim = Sim_data_TIRT(
        npersons=500,
        ntraits=5,
        nblocks_per_trait=12,
        nitems_per_block=2,
        weight_sign=0.5,
        w_range=(0.65, 0.95),
        b_range=(-1.0, 1.0),
        comb_blocks="random",
    ).simulate()

    model = AE_TIRT(
        input_dim=sim.responses.shape[1],
        latent_dim=sim.theta.shape[1],
        item_trait_map=torch.tensor(sim.item_trait_map, dtype=torch.long),
        pair_definitions=torch.tensor(sim.pair_definitions, dtype=torch.long),
        weight_sign=torch.tensor(sim.weight_sign_array, dtype=torch.float32),
        weight_constraint="standardized",
        link_function="probit",
    )

    start = time.time()
    history = train_model(
        model=model,
        train_data=sim.responses,
        optimizer_name="adam",
        batch_size=16,
        num_epochs=500,
        learning_rate=0.001,
        early_stopping_patience=20,
        penalty_weight=sim.responses.shape[1] * 1.0,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    elapsed = time.time() - start

    print(f"Training epochs: {len(history['total_loss'])}")
    print(f"Elapsed seconds: {elapsed:.2f}")


if __name__ == "__main__":
    main()
