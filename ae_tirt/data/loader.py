"""Data loading utilities."""

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
