"""Fit AE-TIRT on external FC data. Input: matrix (persons × pairs) or long-format (person, itemC, response). See data/real/README.md for file formats."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ae_tirt import AE_TIRT, train_model


def _validate_inputs(responses_df, item_trait_df, pair_df, sign_df):
    required_pair_cols = {"item1", "item2"}
    if not required_pair_cols.issubset(set(pair_df.columns)):
        raise ValueError("pair_definitions.csv must include columns: item1,item2")
    if "item_trait" not in item_trait_df.columns:
        raise ValueError("item_trait_map.csv must include column: item_trait")
    if "weight_sign" not in sign_df.columns:
        raise ValueError("weight_sign.csv must include column: weight_sign")

    responses = responses_df.values
    if responses.ndim != 2:
        raise ValueError("responses.csv must be a 2D matrix (rows=persons, cols=pairs).")
    unique_vals = np.unique(responses)
    if not set(unique_vals.tolist()).issubset({0, 1}):
        raise ValueError("responses.csv must contain binary responses only (0/1).")

    npairs = responses.shape[1]
    if len(pair_df) != npairs:
        raise ValueError(
            f"pair_definitions.csv row count ({len(pair_df)}) must equal number of response columns ({npairs})."
        )

    nitems = len(item_trait_df)
    if len(sign_df) != nitems:
        raise ValueError(
            f"weight_sign.csv length ({len(sign_df)}) must equal number of items in item_trait_map.csv ({nitems})."
        )

    item_min = int(pair_df[["item1", "item2"]].min().min())
    item_max = int(pair_df[["item1", "item2"]].max().max())
    if item_min < 1 or item_max > nitems:
        raise ValueError(
            f"pair_definitions.csv contains item indices out of range. Valid item IDs are 1..{nitems}."
        )


def _validate_real_data_shape(item_trait_map: np.ndarray, npairs: int, expect_50_5: bool = False) -> None:
    """Check trait IDs are 1..K; if expect_50_5, require 50 statements, 5 traits, 25 pairs."""
    nitems = len(item_trait_map)
    unique_traits = set(np.unique(item_trait_map).tolist())
    ntraits = int(np.max(item_trait_map))
    if unique_traits != set(range(1, ntraits + 1)):
        raise ValueError(
            f"item_trait_map trait IDs must be consecutive 1..K (got {sorted(unique_traits)})."
        )
    if expect_50_5:
        if nitems != 50 or ntraits != 5:
            raise ValueError(
                f"Strict real-data mode requires 50 statements and 5 traits; got nitems={nitems}, ntraits={ntraits}."
            )
        if npairs != 25:
            raise ValueError(
                f"Strict real-data mode requires 25 pairs (50 statements); got {npairs}."
            )


def _read_item_trait_map(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if {"statement_id", "trait_id"}.issubset(df.columns):
        df = df.sort_values("statement_id")
        return df["trait_id"].to_numpy()
    if "item_trait" in df.columns:
        return df["item_trait"].to_numpy()
    raise ValueError("item_trait_map.csv must have either [item_trait] or [statement_id,trait_id] columns.")


def _read_pair_definitions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"item1", "item2"}.issubset(df.columns):
        return df[["item1", "item2"]]
    if {"statement_j", "statement_k"}.issubset(df.columns):
        return df.rename(columns={"statement_j": "item1", "statement_k": "item2"})[["item1", "item2"]]
    raise ValueError("pair_definitions.csv must have [item1,item2] or [statement_j,statement_k] columns.")


def _read_weight_sign(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if {"statement_id", "weight_sign"}.issubset(df.columns):
        df = df.sort_values("statement_id")
        return df["weight_sign"].to_numpy()
    if "weight_sign" in df.columns:
        return df["weight_sign"].to_numpy()
    raise ValueError("weight_sign.csv must include weight_sign column.")


def _read_responses_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "userid" in df.columns:
        return df.drop(columns=["userid"])
    return df


def _read_responses_from_long(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"person", "itemC", "response"}
    if not required.issubset(df.columns):
        raise ValueError("Real_data.csv must include columns: person,itemC,response")
    wide = df.pivot(index="person", columns="itemC", values="response").sort_index(axis=1)
    wide.columns = [str(c) for c in wide.columns]
    return wide.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Fit AE-TIRT on external forced-choice data.")
    parser.add_argument("--data-dir", type=str, default="data/real", help="Directory containing required CSV files.")
    parser.add_argument(
        "--responses-file",
        type=str,
        default="X_responses.csv",
        help="Matrix response filename (default: X_responses.csv).",
    )
    parser.add_argument(
        "--long-file",
        type=str,
        default="Real_data.csv",
        help="Long-format response filename (default: Real_data.csv).",
    )
    parser.add_argument(
        "--use-long-format",
        action="store_true",
        help="Use long-format Real_data.csv instead of matrix response file.",
    )
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--penalty-weight-factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--weight-constraint", type=str, default="standardized", choices=["standardized", "free"])
    parser.add_argument("--link-function", type=str, default="probit", choices=["probit", "logit"])
    parser.add_argument(
        "--strict-real-data",
        action="store_true",
        help="Enforce paper convention: 50 statements, 5 traits, 25 pairs.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data_dir)
    item_trait_map = _read_item_trait_map(data_dir / "item_trait_map.csv")
    pair_df = _read_pair_definitions(data_dir / "pair_definitions.csv")
    weight_sign = _read_weight_sign(data_dir / "weight_sign.csv")
    responses_df = (
        _read_responses_from_long(data_dir / args.long_file)
        if args.use_long_format
        else _read_responses_matrix(data_dir / args.responses_file)
    )

    _validate_inputs(
        responses_df,
        pd.DataFrame({"item_trait": item_trait_map}),
        pair_df,
        pd.DataFrame({"weight_sign": weight_sign}),
    )

    _validate_real_data_shape(
        item_trait_map,
        npairs=responses_df.values.shape[1],
        expect_50_5=args.strict_real_data,
    )

    responses = responses_df.values
    pair_definitions = pair_df.values

    model = AE_TIRT(
        input_dim=responses.shape[1],
        latent_dim=int(item_trait_map.max()),
        item_trait_map=torch.tensor(item_trait_map, dtype=torch.long),
        pair_definitions=torch.tensor(pair_definitions, dtype=torch.long),
        weight_sign=torch.tensor(weight_sign, dtype=torch.float32),
        weight_constraint=args.weight_constraint,
        link_function=args.link_function,
    )

    device = torch.device(args.device)
    history = train_model(
        model=model,
        train_data=responses,
        optimizer_name=args.optimizer,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        early_stopping_patience=args.early_stopping_patience,
        penalty_weight=responses.shape[1] * args.penalty_weight_factor,
    )
    print("Input validation passed.")
    print(f"Persons={responses.shape[0]}, Pairs={responses.shape[1]}, Items={len(item_trait_map)}")
    print(
        f"Hyperparameters: optimizer={args.optimizer}, batch_size={args.batch_size}, epochs={args.num_epochs}, "
        f"lr={args.learning_rate}, patience={args.early_stopping_patience}, "
        f"penalty_weight_factor={args.penalty_weight_factor}, seed={args.seed}, "
        f"weight_constraint={args.weight_constraint}, link_function={args.link_function}, device={args.device}"
    )
    print(f"Training completed. Epochs: {len(history['total_loss'])}")


if __name__ == "__main__":
    main()
