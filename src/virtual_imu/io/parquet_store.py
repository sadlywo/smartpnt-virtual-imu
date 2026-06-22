"""Parquet read/write helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to Parquet with parent creation."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def read_parquet(path: str | Path) -> pd.DataFrame:
    """Read a Parquet file with a clear missing-file error."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {in_path}")
    return pd.read_parquet(in_path)
