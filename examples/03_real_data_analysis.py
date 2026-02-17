"""Example 3: real-data analysis template."""

from __future__ import annotations

import pandas as pd
import torch

from ae_tirt import AE_TIRT, train_model


def main():
    # Load bundled real data (see data/real/README.md for file provenance).
    resp_df = pd.read_csv("data/real/X_responses.csv")
    if "userid" in resp_df.columns:
        resp_df = resp_df.drop(columns=["userid"])
    responses = resp_df.values

    itm = pd.read_csv("data/real/item_trait_map.csv")
    if "trait_id" in itm.columns:
        itm = itm.sort_values("statement_id")
        item_trait_map = itm["trait_id"].values
    else:
        item_trait_map = itm["item_trait"].values

    pair_df = pd.read_csv("data/real/pair_definitions.csv")
    if "statement_j" in pair_df.columns:
        pair_df = pair_df.rename(columns={"statement_j": "item1", "statement_k": "item2"})
    pair_definitions = pair_df[["item1", "item2"]].values

    sign_df = pd.read_csv("data/real/weight_sign.csv")
    if "statement_id" in sign_df.columns:
        sign_df = sign_df.sort_values("statement_id")
    weight_sign = sign_df["weight_sign"].values

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
