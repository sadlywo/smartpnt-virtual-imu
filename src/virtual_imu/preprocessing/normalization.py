"""Feature normalization utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_standardization(df: pd.DataFrame, columns: list[str]) -> dict[str, list[float]]:
    """Compute mean/std for selected columns."""
    values = df[columns].to_numpy(dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"columns": columns, "mean": mean.tolist(), "std": std.tolist()}


def apply_standardization(values: np.ndarray, stats: dict[str, list[float]], inverse: bool = False) -> np.ndarray:
    """Apply or invert standardization to arrays with channel-last layout."""
    mean = np.asarray(stats["mean"], dtype=float)
    std = np.asarray(stats["std"], dtype=float)
    return values * std + mean if inverse else (values - mean) / std
