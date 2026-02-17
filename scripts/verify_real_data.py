"""Verify data/real: 5 traits, 50 statements, 25 pairs; AE-TIRT files consistent with items_2MFC. See data/real/README.md."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data" / "real"
    if not data_dir.is_dir():
        print("data/real not found.", file=sys.stderr)
        sys.exit(1)

    errors = []

    # 1) item_trait_map: 50 rows, trait_id 1..5
    itm_path = data_dir / "item_trait_map.csv"
    itm = pd.read_csv(itm_path)
    if {"statement_id", "trait_id"}.issubset(itm.columns):
        itm = itm.sort_values("statement_id")
        trait_arr = itm["trait_id"].to_numpy()
    elif "item_trait" in itm.columns:
        trait_arr = itm["item_trait"].to_numpy()
    else:
        errors.append("item_trait_map.csv must have statement_id,trait_id or item_trait")
        trait_arr = np.array([])

    nitems = len(trait_arr)
    if nitems != 50:
        errors.append(f"item_trait_map: expected 50 statements, got {nitems}")
    unique_traits = set(np.unique(trait_arr).tolist())
    if unique_traits != {1, 2, 3, 4, 5}:
        errors.append(f"item_trait_map: trait_id must be 1..5, got {sorted(unique_traits)}")

    # 2) pair_definitions: 25 rows, statement IDs 1..50
    pair_path = data_dir / "pair_definitions.csv"
    pair_df = pd.read_csv(pair_path)
    if {"statement_j", "statement_k"}.issubset(pair_df.columns):
        pair_df = pair_df.rename(columns={"statement_j": "item1", "statement_k": "item2"})
    if "item1" not in pair_df.columns or "item2" not in pair_df.columns:
        errors.append("pair_definitions.csv must have item1,item2 or statement_j,statement_k")
    else:
        npairs = len(pair_df)
        if npairs != 25:
            errors.append(f"pair_definitions: expected 25 pairs, got {npairs}")
        imin = int(pair_df[["item1", "item2"]].min().min())
        imax = int(pair_df[["item1", "item2"]].max().max())
        if imin < 1 or imax > 50:
            errors.append(f"pair_definitions: statement IDs must be 1..50, got range {imin}..{imax}")

    # 3) weight_sign: 50 rows
    sign_path = data_dir / "weight_sign.csv"
    sign_df = pd.read_csv(sign_path)
    if "weight_sign" in sign_df.columns:
        if "statement_id" in sign_df.columns:
            sign_df = sign_df.sort_values("statement_id")
        sign_arr = sign_df["weight_sign"].to_numpy()
    else:
        errors.append("weight_sign.csv must have weight_sign column")
        sign_arr = np.array([])
    if len(sign_arr) != 50:
        errors.append(f"weight_sign: expected 50 rows, got {len(sign_arr)}")

    # 4) Cross-trait pairs (each pair must be two different traits)
    if len(trait_arr) == 50 and "item1" in pair_df.columns:
        for idx, row in pair_df.iterrows():
            j, k = int(row["item1"]), int(row["item2"])
            if j < 1 or k < 1 or j > 50 or k > 50:
                continue
            t1, t2 = trait_arr[j - 1], trait_arr[k - 1]
            if t1 == t2:
                errors.append(f"Pair {idx + 1} (statements {j},{k}) same trait {t1}; FC requires cross-trait.")

    # 5) Consistency with items_2MFC.csv (first 25 rows -> 50 statements)
    items_path = data_dir / "items_2MFC.csv"
    if items_path.exists() and len(trait_arr) == 50 and len(sign_arr) == 50:
        items = pd.read_csv(items_path).head(25)
        expected_trait = []
        expected_sign = []
        for _, r in items.iterrows():
            expected_trait.extend([r["M1"], r["M2"]])
            expected_sign.extend([float(r["R1"]), float(r["R2"])])
        if not np.array_equal(expected_trait, trait_arr):
            errors.append("item_trait_map trait_id does not match items_2MFC.csv (M1,M2 first 25 rows).")
        if not np.array_equal(expected_sign, sign_arr):
            errors.append("weight_sign does not match items_2MFC.csv (R1,R2 first 25 rows).")

    # 6) Response matrix: 25 columns
    resp_path = data_dir / "X_responses.csv"
    if resp_path.exists():
        resp = pd.read_csv(resp_path)
        if "userid" in resp.columns:
            resp = resp.drop(columns=["userid"])
        ncols = resp.shape[1]
        if ncols != 25:
            errors.append(f"X_responses.csv: expected 25 pair columns, got {ncols}")

    if errors:
        for e in errors:
            print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
    print("OK: 5 traits, 50 statements, 25 pairs; trait and sign mapping consistent with items_2MFC (first 25 rows).")


if __name__ == "__main__":
    main()
