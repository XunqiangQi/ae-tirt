"""Example 3: real-data analysis template."""

from __future__ import annotations

import pandas as pd
import torch

from ae_tirt import AE_TIRT, train_model


def main():
    # Replace with your own file paths.
    responses = pd.read_csv("data/real/responses.csv").values
    item_trait_map = pd.read_csv("data/real/item_trait_map.csv")["item_trait"].values
    pair_definitions = pd.read_csv("data/real/pair_definitions.csv")[["item1", "item2"]].values
    weight_sign = pd.read_csv("data/real/weight_sign.csv")["weight_sign"].values

    model = AE_TIRT(
        input_dim=responses.shape[1],
        latent_dim=int(item_trait_map.max()),
        item_trait_map=torch.tensor(item_trait_map, dtype=torch.long),
        pair_definitions=torch.tensor(pair_definitions, dtype=torch.long),
        weight_sign=torch.tensor(weight_sign, dtype=torch.float32),
        weight_constraint="standardized",
        link_function="probit",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(
        model=model,
        train_data=responses,
        optimizer_name="adam",
        batch_size=16,
        num_epochs=500,
        learning_rate=1e-3,
        device=device,
        early_stopping_patience=20,
        penalty_weight=responses.shape[1] * 1.0,
    )

    # If external criterion traits exist, pass them into evaluate_model.
    # metrics = evaluate_model(model, responses, true_traits, device=device)
    # print(metrics["traits"]["overall"])
    print("Model training finished on real data.")


if __name__ == "__main__":
    main()
