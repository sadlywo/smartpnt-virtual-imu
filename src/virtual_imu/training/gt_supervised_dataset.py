"""GT-supervised SmartPNT-MSF PyTorch dataloaders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from virtual_imu.preprocessing.normalization import apply_standardization, compute_standardization
from virtual_imu.preprocessing.windowing import make_window_slices

SMARTPNT_GT_IMU_COLUMNS = [
    "gx_radps",
    "gy_radps",
    "gz_radps",
    "ax_mps2",
    "ay_mps2",
    "az_mps2",
]
SMARTPNT_GT_VELOCITY_COLUMNS = [
    "gt_v_east_mps",
    "gt_v_north_mps",
    "gt_v_up_mps",
]
SMARTPNT_GT_POSITION_COLUMNS = [
    "gt_x_local_m",
    "gt_y_local_m",
    "gt_z_local_m",
]
SMARTPNT_GT_ATTITUDE_COLUMNS = [
    "gt_heading_deg",
    "gt_pitch_deg",
    "gt_roll_deg",
]
SMARTPNT_GT_POSE_DELTA_COLUMNS = [
    "delta_x_local_m",
    "delta_y_local_m",
    "delta_z_local_m",
    "delta_heading_deg",
    "delta_pitch_deg",
    "delta_roll_deg",
]
SMARTPNT_GT_BODY_POSE_DELTA_COLUMNS = [
    "delta_forward_m",
    "delta_right_m",
    "delta_up_m",
    "delta_heading_deg",
    "delta_pitch_deg",
    "delta_roll_deg",
]
SMARTPNT_GT_POSITION_DELTA_COLUMNS = [
    "delta_x_local_m",
    "delta_y_local_m",
    "delta_z_local_m",
]
SMARTPNT_GT_BODY_POSITION_DELTA_COLUMNS = [
    "delta_forward_m",
    "delta_right_m",
    "delta_up_m",
]
SMARTPNT_GT_SEQUENCE_LABELS = {
    "velocity": SMARTPNT_GT_VELOCITY_COLUMNS,
    "position": SMARTPNT_GT_POSITION_COLUMNS,
    "velocity_position": [*SMARTPNT_GT_VELOCITY_COLUMNS, *SMARTPNT_GT_POSITION_COLUMNS],
}
SMARTPNT_GT_VECTOR_LABELS = {
    "position_delta": SMARTPNT_GT_POSITION_DELTA_COLUMNS,
    "pose_delta": SMARTPNT_GT_POSE_DELTA_COLUMNS,
    "body_position_delta": SMARTPNT_GT_BODY_POSITION_DELTA_COLUMNS,
    "body_pose_delta": SMARTPNT_GT_BODY_POSE_DELTA_COLUMNS,
}


def _as_frame(data: pd.DataFrame | str | Path | list[str | Path]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.is_dir():
            paths = sorted(path.glob("*_imu_gt_aligned.parquet"))
            if not paths:
                raise FileNotFoundError(f"No aligned SmartPNT GT parquet files found under {path}")
            return pd.concat([pd.read_parquet(item) for item in paths], ignore_index=True)
        return pd.read_parquet(path)
    if not data:
        raise ValueError("At least one SmartPNT GT parquet path is required")
    return pd.concat([pd.read_parquet(Path(item)) for item in data], ignore_index=True)


def _stats_from_values(values: np.ndarray, columns: list[str]) -> dict[str, list[float]]:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {"columns": columns, "mean": mean.tolist(), "std": std.tolist()}


def _wrap_degrees(delta: np.ndarray) -> np.ndarray:
    return (delta + 180.0) % 360.0 - 180.0


def _heading_aligned_delta(dx: float, dy: float, dz: float, heading_deg: float) -> np.ndarray:
    heading = np.deg2rad(float(heading_deg))
    forward = np.sin(heading) * dx + np.cos(heading) * dy
    right = np.cos(heading) * dx - np.sin(heading) * dy
    return np.asarray([forward, right, dz], dtype=np.float32)


def _integrate_trapezoid(values: np.ndarray, timestamps: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.trapz(values.astype(float), timestamps.astype(float)))


class SmartPNTGTWindowDataset(Dataset):
    """Windowed SmartPNT-MSF dataset for inertial odometry supervision."""

    def __init__(
        self,
        data: pd.DataFrame | str | Path | list[str | Path],
        window_size: int,
        stride: int,
        label_mode: str = "velocity_position",
        feature_stats: dict[str, list[float]] | None = None,
        target_stats: dict[str, list[float]] | None = None,
        normalize_features: bool = True,
        normalize_targets: bool = True,
    ) -> None:
        self.label_mode = label_mode
        if self.label_mode not in {*SMARTPNT_GT_SEQUENCE_LABELS, *SMARTPNT_GT_VECTOR_LABELS}:
            valid = sorted([*SMARTPNT_GT_SEQUENCE_LABELS, *SMARTPNT_GT_VECTOR_LABELS])
            raise ValueError(f"Unsupported SmartPNT GT label_mode: {self.label_mode}. Valid modes: {valid}")
        self.df = self._ensure_columns(_as_frame(data))
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.normalize_features = bool(normalize_features)
        self.normalize_targets = bool(normalize_targets)
        self.target_columns = self._target_columns()
        self.target_layout = "sequence" if self.label_mode in SMARTPNT_GT_SEQUENCE_LABELS else "window"
        self.feature_stats = feature_stats or (
            compute_standardization(self.df, SMARTPNT_GT_IMU_COLUMNS) if self.normalize_features else None
        )
        self.windows: list[dict[str, Any]] = []
        target_values_for_stats = []
        for sequence_id, group in self.df.groupby("sequence_id", sort=False):
            group = group.sort_values("timestamp_s").reset_index(drop=True)
            features = group[SMARTPNT_GT_IMU_COLUMNS].to_numpy(dtype=np.float32)
            if self.normalize_features and self.feature_stats is not None:
                features = apply_standardization(features, self.feature_stats).astype(np.float32)
            for slc in make_window_slices(len(group), self.window_size, self.stride):
                window_frame = group.iloc[slc]
                target = self._target_values(window_frame).astype(np.float32)
                integration = self._integration_targets(window_frame)
                target_values_for_stats.append(target if target.ndim == 2 else target[None, :])
                self.windows.append(
                    {
                        "x": features[slc],
                        "y": target,
                        "sequence_id": str(sequence_id),
                        "start_index": int(slc.start or 0),
                        "end_index": int((slc.stop or 0) - 1),
                        "start_time_s": float(group.iloc[slc.start or 0]["timestamp_s"]),
                        "end_time_s": float(group.iloc[(slc.stop or 0) - 1]["timestamp_s"]),
                        **integration,
                    }
                )
        if not self.windows:
            raise ValueError(
                f"No SmartPNT GT windows created; samples={len(self.df)}, window_size={self.window_size}, stride={self.stride}"
            )
        self.target_stats = target_stats or (
            _stats_from_values(np.concatenate(target_values_for_stats, axis=0), self.target_columns)
            if self.normalize_targets
            else None
        )
        if self.normalize_targets and self.target_stats is not None:
            for window in self.windows:
                y = window["y"]
                if y.ndim == 1:
                    window["y"] = apply_standardization(y[None, :], self.target_stats).astype(np.float32)[0]
                else:
                    window["y"] = apply_standardization(y, self.target_stats).astype(np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        window = self.windows[index]
        x = torch.from_numpy(np.ascontiguousarray(window["x"].T))
        y_values = window["y"]
        y = torch.from_numpy(
            np.ascontiguousarray(y_values.T if self.target_layout == "sequence" else y_values)
        )
        meta = {
            "sequence_id": window["sequence_id"],
            "start_index": window["start_index"],
            "end_index": window["end_index"],
            "start_time_s": window["start_time_s"],
            "end_time_s": window["end_time_s"],
            "label_mode": self.label_mode,
            "target_layout": self.target_layout,
            "feature_columns": SMARTPNT_GT_IMU_COLUMNS,
            "target_columns": self.target_columns,
            "integrated_delta_x_local_m": window["integrated_delta_x_local_m"],
            "integrated_delta_y_local_m": window["integrated_delta_y_local_m"],
            "integrated_delta_z_local_m": window["integrated_delta_z_local_m"],
            "integrated_delta_forward_m": window["integrated_delta_forward_m"],
            "integrated_delta_right_m": window["integrated_delta_right_m"],
            "integrated_delta_up_m": window["integrated_delta_up_m"],
            "integrated_delta_heading_deg": window["integrated_delta_heading_deg"],
            "gyro_integrated_delta_heading_deg": window["gyro_integrated_delta_heading_deg"],
        }
        return x, y, meta

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "sequence_id" not in out.columns:
            out["sequence_id"] = "sequence"
        required = [
            "timestamp_s",
            "sequence_id",
            *SMARTPNT_GT_IMU_COLUMNS,
            *SMARTPNT_GT_VELOCITY_COLUMNS,
            *SMARTPNT_GT_POSITION_COLUMNS,
        ]
        if self._requires_attitude():
            required.extend(SMARTPNT_GT_ATTITUDE_COLUMNS)
        missing = [column for column in required if column not in out.columns]
        if missing:
            raise ValueError(f"Missing required SmartPNT GT columns: {missing}")
        return out

    def _requires_attitude(self) -> bool:
        return self.label_mode in {"pose_delta", "body_position_delta", "body_pose_delta"}

    def _target_columns(self) -> list[str]:
        if self.label_mode in SMARTPNT_GT_SEQUENCE_LABELS:
            return list(SMARTPNT_GT_SEQUENCE_LABELS[self.label_mode])
        return list(SMARTPNT_GT_VECTOR_LABELS[self.label_mode])

    def _target_values(self, window: pd.DataFrame) -> np.ndarray:
        if self.label_mode in SMARTPNT_GT_SEQUENCE_LABELS:
            return window[self.target_columns].to_numpy(dtype=np.float32)
        start = window.iloc[0]
        end = window.iloc[-1]
        position_delta = self._position_delta(start, end)
        if self.label_mode == "position_delta":
            return position_delta.astype(np.float32)
        if self.label_mode == "body_position_delta":
            return self._body_position_delta(start, end).astype(np.float32)
        attitude_delta = (
            end[SMARTPNT_GT_ATTITUDE_COLUMNS].to_numpy(dtype=np.float32)
            - start[SMARTPNT_GT_ATTITUDE_COLUMNS].to_numpy(dtype=np.float32)
        )
        attitude_delta = _wrap_degrees(attitude_delta)
        if self.label_mode == "body_pose_delta":
            return np.concatenate([self._body_position_delta(start, end), attitude_delta]).astype(np.float32)
        return np.concatenate([position_delta, attitude_delta]).astype(np.float32)

    def _position_delta(self, start: pd.Series, end: pd.Series) -> np.ndarray:
        return (
            end[SMARTPNT_GT_POSITION_COLUMNS].to_numpy(dtype=np.float32)
            - start[SMARTPNT_GT_POSITION_COLUMNS].to_numpy(dtype=np.float32)
        )

    def _body_position_delta(self, start: pd.Series, end: pd.Series) -> np.ndarray:
        dx, dy, dz = self._position_delta(start, end)
        return _heading_aligned_delta(float(dx), float(dy), float(dz), float(start["gt_heading_deg"]))

    def _integration_targets(self, window: pd.DataFrame) -> dict[str, float]:
        timestamps = window["timestamp_s"].to_numpy(dtype=float)
        dx = _integrate_trapezoid(window["gt_v_east_mps"].to_numpy(dtype=float), timestamps)
        dy = _integrate_trapezoid(window["gt_v_north_mps"].to_numpy(dtype=float), timestamps)
        dz = _integrate_trapezoid(window["gt_v_up_mps"].to_numpy(dtype=float), timestamps)
        body = _heading_aligned_delta(dx, dy, dz, float(window.iloc[0]["gt_heading_deg"]))
        heading_values = window["gt_heading_deg"].to_numpy(dtype=float)
        heading_steps = _wrap_degrees(np.diff(heading_values))
        integrated_heading = float(np.sum(heading_steps)) if len(heading_steps) else 0.0
        gyro_heading = _integrate_trapezoid(window["gz_radps"].to_numpy(dtype=float), timestamps) * 180.0 / np.pi
        return {
            "integrated_delta_x_local_m": float(dx),
            "integrated_delta_y_local_m": float(dy),
            "integrated_delta_z_local_m": float(dz),
            "integrated_delta_forward_m": float(body[0]),
            "integrated_delta_right_m": float(body[1]),
            "integrated_delta_up_m": float(body[2]),
            "integrated_delta_heading_deg": integrated_heading,
            "gyro_integrated_delta_heading_deg": float(gyro_heading),
        }


def make_smartpnt_gt_dataloader(
    data: pd.DataFrame | str | Path | list[str | Path],
    window_size: int,
    stride: int,
    batch_size: int,
    label_mode: str = "velocity_position",
    feature_stats: dict[str, list[float]] | None = None,
    target_stats: dict[str, list[float]] | None = None,
    normalize_features: bool = True,
    normalize_targets: bool = True,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader over aligned SmartPNT-MSF IMU/GT parquet data."""
    dataset = SmartPNTGTWindowDataset(
        data,
        window_size=window_size,
        stride=stride,
        label_mode=label_mode,
        feature_stats=feature_stats,
        target_stats=target_stats,
        normalize_features=normalize_features,
        normalize_targets=normalize_targets,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        drop_last=bool(drop_last),
    )


def make_smartpnt_gt_dataloaders(
    train_data: pd.DataFrame | str | Path | list[str | Path],
    val_data: pd.DataFrame | str | Path | list[str | Path],
    window_size: int,
    stride: int,
    batch_size: int,
    label_mode: str = "velocity_position",
    normalize_features: bool = True,
    normalize_targets: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create train/validation loaders sharing train-set normalization stats."""
    train_ds = SmartPNTGTWindowDataset(
        train_data,
        window_size=window_size,
        stride=stride,
        label_mode=label_mode,
        normalize_features=normalize_features,
        normalize_targets=normalize_targets,
    )
    val_ds = SmartPNTGTWindowDataset(
        val_data,
        window_size=window_size,
        stride=stride,
        label_mode=label_mode,
        feature_stats=train_ds.feature_stats,
        target_stats=train_ds.target_stats,
        normalize_features=normalize_features,
        normalize_targets=normalize_targets,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
    )
    return train_loader, val_loader


def _sequence_subset(df: pd.DataFrame, sequences: list[str], split_name: str) -> pd.DataFrame:
    if not sequences:
        raise ValueError(f"SmartPNT GT split '{split_name}' has no sequences")
    identity_column = "original_sequence_id" if "original_sequence_id" in df else "sequence_id"
    available = set(df[identity_column].astype(str).unique())
    requested = [str(sequence) for sequence in sequences]
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"SmartPNT GT split '{split_name}' requested missing sequences: {missing}")
    return df[df[identity_column].astype(str).isin(requested)].reset_index(drop=True)


def make_smartpnt_gt_split_dataloaders(
    data: pd.DataFrame | str | Path | list[str | Path],
    split_sequences: dict[str, list[str]],
    window_size: int,
    stride: int,
    batch_size: int,
    label_mode: str = "velocity_position",
    normalize_features: bool = True,
    normalize_targets: bool = True,
    num_workers: int = 0,
    skipped_sequences: list[str] | None = None,
) -> dict[str, DataLoader]:
    """Create train/val/test DataLoaders from explicit SmartPNT sequence splits."""
    if "train" not in split_sequences:
        raise ValueError("split_sequences must include a 'train' split")
    df = _as_frame(data)
    if "sequence_id" not in df.columns:
        raise ValueError("SmartPNT GT sequence splits require a sequence_id column")
    if skipped_sequences:
        skipped = {str(sequence) for sequence in skipped_sequences}
        df = df[~df["sequence_id"].astype(str).isin(skipped)].reset_index(drop=True)

    train_frame = _sequence_subset(df, split_sequences["train"], "train")
    train_ds = SmartPNTGTWindowDataset(
        train_frame,
        window_size=window_size,
        stride=stride,
        label_mode=label_mode,
        normalize_features=normalize_features,
        normalize_targets=normalize_targets,
    )
    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=int(num_workers),
        )
    }
    for split_name in ("val", "test"):
        if split_name not in split_sequences:
            continue
        split_frame = _sequence_subset(df, split_sequences[split_name], split_name)
        split_ds = SmartPNTGTWindowDataset(
            split_frame,
            window_size=window_size,
            stride=stride,
            label_mode=label_mode,
            feature_stats=train_ds.feature_stats,
            target_stats=train_ds.target_stats,
            normalize_features=normalize_features,
            normalize_targets=normalize_targets,
        )
        loaders[split_name] = DataLoader(
            split_ds,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=int(num_workers),
        )
    return loaders
