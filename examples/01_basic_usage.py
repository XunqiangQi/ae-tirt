"""Example 1: basic end-to-end AE-TIRT workflow on synthetic data."""

import torch

from ae_tirt import AE_TIRT, Sim_data_TIRT, evaluate_model, train_model


def main():
    sim = Sim_data_TIRT(
        npersons=300,
        ntraits=5,
        nblocks_per_trait=6,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = train_model(
        model=model,
        train_data=sim.responses,
        optimizer_name="adam",
        batch_size=64,
        num_epochs=50,
        learning_rate=1e-3,
        device=device,
        early_stopping_patience=10,
        penalty_weight=sim.responses.shape[1] * 0.1,
    )
    metrics = evaluate_model(model, sim.responses, sim.theta, device=device)

    print(f"Trained epochs: {len(history['total_loss'])}")
    print(f"Overall trait RMSE: {metrics['traits']['overall']['rmse']:.4f}")
    print(f"Overall trait correlation: {metrics['traits']['overall']['cor']:.4f}")


if __name__ == "__main__":
    main()
