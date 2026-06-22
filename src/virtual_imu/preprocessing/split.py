"""Dataset splitting helpers."""

from __future__ import annotations

import pandas as pd


def split_by_fraction(
    df: pd.DataFrame,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Split a time-ordered DataFrame into train/val/test fractions."""
    total = train + val + test
    if total <= 0:
        raise ValueError("Split fractions must sum to a positive value")
    train_n = int(len(df) * train / total)
    val_n = int(len(df) * val / total)
    return {
        "train": df.iloc[:train_n].copy(),
        "val": df.iloc[train_n : train_n + val_n].copy(),
        "test": df.iloc[train_n + val_n :].copy(),
    }
