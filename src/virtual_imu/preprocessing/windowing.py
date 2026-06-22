"""Windowing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_window_slices(n_samples: int, window_size: int, stride: int) -> list[slice]:
    """Return window slices with strict positive size and stride."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if n_samples < window_size:
        return []
    return [slice(start, start + window_size) for start in range(0, n_samples - window_size + 1, stride)]


def window_long_dataframe(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    target: str = "full_sequence",
) -> pd.DataFrame:
    """Create a long-format windowed DataFrame preserving sample rows and adding window_id."""
    rows: list[pd.DataFrame] = []
    for sequence_id, group in df.groupby("sequence_id", sort=False):
        group = group.sort_values("timestamp_s").reset_index(drop=True)
        for window_idx, window_slice in enumerate(make_window_slices(len(group), window_size, stride)):
            window = group.iloc[window_slice].copy()
            if target == "center":
                center = len(window) // 2
                window = window.iloc[[center]].copy()
            window["window_id"] = f"{sequence_id}_{window_idx:06d}"
            window["sample_in_window"] = np.arange(len(window))
            rows.append(window)
    if not rows:
        return pd.DataFrame(columns=[*df.columns, "window_id", "sample_in_window"])
    return pd.concat(rows, ignore_index=True)
