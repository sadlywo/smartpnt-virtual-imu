"""Coordinate and axis mapping helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_axis_mapping(df: pd.DataFrame, columns: list[str], axis_mapping: list[str], sign_mapping: list[int]) -> pd.DataFrame:
    """Apply axis order and sign mapping to a 3-axis vector column group."""
    if len(columns) != 3 or len(axis_mapping) != 3 or len(sign_mapping) != 3:
        raise ValueError("columns, axis_mapping, and sign_mapping must have length 3")
    axis_to_idx = {"x": 0, "y": 1, "z": 2}
    values = df[columns].to_numpy(dtype=float)
    mapped = np.column_stack([sign_mapping[i] * values[:, axis_to_idx[axis_mapping[i]]] for i in range(3)])
    out = df.copy()
    out[columns] = mapped
    return out
