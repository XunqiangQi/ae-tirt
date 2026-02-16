"""I/O helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_csv(df: pd.DataFrame, path: str | Path, **kwargs) -> None:
    """Save a DataFrame to CSV (index=False by default)."""
    pd.DataFrame(df).to_csv(path, index=False, **kwargs)
