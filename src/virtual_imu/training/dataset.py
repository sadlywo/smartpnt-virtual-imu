"""PyTorch datasets for IMU windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from virtual_imu.preprocessing.normalization import apply_standardization, compute_standardization
from virtual_imu.preprocessing.windowing import make_window_slices

FEATURE_COLUMNS = ["gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low", "temperature_c"]
TARGET_COLUMNS = ["gx_high", "gy_high", "gz_high", "ax_high", "ay_high", "az_high"]
LOW_IMU_COLUMNS = ["gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low"]
COMPETITOR_COLUMNS = [
    "gx_competitor",
    "gy_competitor",
    "gz_competitor",
    "ax_competitor",
    "ay_competitor",
    "az_competitor",
]


def ensure_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required feature/target columns exist with sensible defaults."""
    out = df.copy()
    if "temperature_c" not in out.columns:
        out["temperature_c"] = 25.0
    missing = [col for col in [*FEATURE_COLUMNS[:6], *TARGET_COLUMNS] if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required IMU columns: {missing}")
    if "sequence_id" not in out.columns:
        out["sequence_id"] = "sequence"
    return out


class IMUWindowDataset(Dataset):
    """Windowed IMU sequence dataset with channel-first tensors."""

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int,
        stride: int,
        feature_stats: dict | None = None,
        target_stats: dict | None = None,
        target_mode: str = "direct",
        competitor_columns: list[str] | None = None,
        zero_target_mean: bool = False,
    ) -> None:
        self.df = ensure_training_columns(df)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.target_mode = target_mode
        self.competitor_columns = competitor_columns
        if self.target_mode not in {"direct", "residual", "competitor_residual"}:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")
        if self.target_mode == "competitor_residual" and self.competitor_columns is None:
            raise ValueError("target_mode='competitor_residual' requires competitor baseline columns")
        if self.competitor_columns is not None:
            missing_competitor = [col for col in self.competitor_columns if col not in self.df.columns]
            if missing_competitor:
                raise ValueError(f"Missing competitor baseline columns: {missing_competitor}")
        self.feature_stats = feature_stats or compute_standardization(self.df, FEATURE_COLUMNS)
        target_values = self._target_values(self.df)
        target_frame = pd.DataFrame(target_values, columns=TARGET_COLUMNS)
        self.target_stats = target_stats or compute_standardization(target_frame, TARGET_COLUMNS)
        if target_stats is None and zero_target_mean:
            self.target_stats = {**self.target_stats, "mean": [0.0] * len(TARGET_COLUMNS)}
        self.windows: list[tuple[np.ndarray, ...]] = []
        for _, group in self.df.groupby("sequence_id", sort=False):
            group = group.sort_values("timestamp_s").reset_index(drop=True)
            features = group[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
            targets = self._target_values(group).astype(np.float32)
            competitors = self._competitor_values(group).astype(np.float32) if self.competitor_columns else None
            physical_competitors = (
                group[self.competitor_columns].to_numpy(dtype=np.float32) if self.competitor_columns else None
            )
            features = apply_standardization(features, self.feature_stats).astype(np.float32)
            targets = apply_standardization(targets, self.target_stats).astype(np.float32)
            if competitors is not None:
                competitors = apply_standardization(competitors, self.target_stats).astype(np.float32)
            for slc in make_window_slices(len(group), self.window_size, self.stride):
                if competitors is None:
                    self.windows.append((features[slc], targets[slc]))
                else:
                    assert physical_competitors is not None
                    self.windows.append((features[slc], targets[slc], competitors[slc], physical_competitors[slc]))
        if not self.windows:
            raise ValueError(
                f"No windows created; samples={len(self.df)}, window_size={self.window_size}, stride={self.stride}"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        window = self.windows[index]
        x = torch.from_numpy(np.ascontiguousarray(window[0].T))
        y = torch.from_numpy(np.ascontiguousarray(window[1].T))
        if len(window) == 2:
            return x, y
        competitor = torch.from_numpy(np.ascontiguousarray(window[2].T))
        physical_competitor = torch.from_numpy(np.ascontiguousarray(window[3].T))
        return x, y, competitor, physical_competitor

    def _target_values(self, df: pd.DataFrame) -> np.ndarray:
        high = df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        if self.target_mode == "direct":
            return high
        if self.target_mode == "residual":
            low = df[LOW_IMU_COLUMNS].to_numpy(dtype=np.float32)
            return high - low
        if self.competitor_columns is None:
            raise RuntimeError("competitor_residual target requested without competitor columns")
        competitor = df[self.competitor_columns].to_numpy(dtype=np.float32)
        return high - competitor

    def _competitor_values(self, df: pd.DataFrame) -> np.ndarray:
        if self.competitor_columns is None:
            raise RuntimeError("Competitor columns requested before they were configured")
        competitor = df[self.competitor_columns].to_numpy(dtype=np.float32)
        if self.target_mode == "direct":
            return competitor
        if self.target_mode == "residual":
            low = df[LOW_IMU_COLUMNS].to_numpy(dtype=np.float32)
            return competitor - low
        return np.zeros_like(competitor, dtype=np.float32)
