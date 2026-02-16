"""I/O helpers."""

import pandas as pd


def save_csv(df, path):
    pd.DataFrame(df).to_csv(path, index=False)
