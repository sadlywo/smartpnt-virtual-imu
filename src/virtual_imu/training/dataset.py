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
STATIC_CONTEXT_COLUMNS = [*LOW_IMU_COLUMNS, "temperature_c"]
TRAIN_GLOBAL_STATIC_CONTEXT_KEY = "__train_global__"
COMPETITOR_COLUMNS = [
    "gx_competitor",
    "gy_competitor",
    "gz_competitor",
    "ax_competitor",
    "ay_competitor",
    "az_competitor",
]


def ensure_training_columns(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Ensure required feature/target columns exist with sensible defaults."""
    out = df.copy()
    if "training_eligible" in out.columns:
        eligibility = out["training_eligible"].astype("boolean").fillna(False)
        if not eligibility.all():
            raise ValueError("Input contains rows that are not eligible for training")
    if "temperature_c" not in out.columns:
        out["temperature_c"] = 25.0
    features = list(feature_columns or FEATURE_COLUMNS)
    missing = [col for col in [*features, *TARGET_COLUMNS] if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required IMU columns: {missing}")
    if "sequence_id" not in out.columns:
        out["sequence_id"] = "sequence"
    return out


def static_context_enabled(static_context: dict | None) -> bool:
    """Return whether a static-context side input is configured."""
    return bool(static_context and static_context.get("enabled", False))


def _static_columns(static_context: dict | None) -> list[str]:
    return list((static_context or {}).get("columns", STATIC_CONTEXT_COLUMNS))


def _static_group_key(group: pd.DataFrame) -> str:
    if "low_sensor" in group.columns:
        return str(group["low_sensor"].iloc[0])
    if "sequence_id" in group.columns:
        return str(group["sequence_id"].iloc[0])
    return "sequence"


def _longest_static_source(group: pd.DataFrame) -> pd.DataFrame:
    if "motion_label" in group.columns and "sequence_id" in group.columns:
        stationary = group[group["motion_label"].astype(str).str.lower().eq("stationary")]
        if not stationary.empty:
            lengths = stationary.groupby("sequence_id", sort=False).size()
            return stationary[stationary["sequence_id"].eq(lengths.idxmax())]
    if "motion_label" in group.columns:
        stationary = group[group["motion_label"].astype(str).str.lower().eq("stationary")]
        if not stationary.empty:
            return stationary
    return group


def _fixed_length_window(values: np.ndarray, window_size: int) -> np.ndarray:
    if len(values) >= window_size:
        return values[:window_size]
    if len(values) == 0:
        raise ValueError("Cannot build static context from an empty source")
    pad = window_size - len(values)
    return np.pad(values, ((0, pad), (0, 0)), mode="edge")


def _rolling_window_mean(values: np.ndarray, window_size: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    return (cumulative[window_size:] - cumulative[:-window_size]) / float(window_size)


def _robust_positive_scale(values: np.ndarray) -> float:
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size == 0:
        return 1.0
    return max(float(np.median(finite)), np.finfo(np.float64).eps)


def select_low_motion_source(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Select one deterministic low-motion window without inspecting high-side targets."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    missing = [column for column in [*LOW_IMU_COLUMNS, "sequence_id"] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing low-motion selection columns: {missing}")

    candidates: list[tuple[float, float, tuple[str, ...], int]] = []
    ordered_sequences: dict[tuple[str, ...], pd.DataFrame] = {}
    group_columns = ["low_sensor", "sequence_id"] if "low_sensor" in df.columns else ["sequence_id"]
    for group_key, sequence in df.groupby(group_columns, sort=True):
        ordered = sequence.sort_values("timestamp_s", kind="stable")
        if len(ordered) < window_size:
            continue
        raw_key = group_key if isinstance(group_key, tuple) else (group_key,)
        sequence_key = tuple(str(value) for value in raw_key)
        ordered_sequences[sequence_key] = ordered
        low = ordered[LOW_IMU_COLUMNS].to_numpy(dtype=np.float64)
        gyro_energy = np.sum(np.square(low[:, :3]), axis=1)
        accel_norm = np.linalg.norm(low[:, 3:], axis=1)
        gyro_mean = _rolling_window_mean(gyro_energy, window_size)
        accel_mean = _rolling_window_mean(accel_norm, window_size)
        accel_sq_mean = _rolling_window_mean(np.square(accel_norm), window_size)
        accel_variance = np.maximum(accel_sq_mean - np.square(accel_mean), 0.0)
        for start in range(len(gyro_mean)):
            gyro_score = gyro_mean[start]
            accel_score = accel_variance[start]
            candidates.append((float(gyro_score), float(accel_score), sequence_key, start))

    if not candidates:
        raise ValueError(f"No sequence contains the requested {window_size}-sample static context window")
    gyro_values = np.asarray([item[0] for item in candidates], dtype=np.float64)
    accel_values = np.asarray([item[1] for item in candidates], dtype=np.float64)
    gyro_scale = _robust_positive_scale(gyro_values)
    accel_scale = _robust_positive_scale(accel_values)
    ranked = [
        (item[0] / gyro_scale + item[1] / accel_scale, item[2], item[3])
        for item in candidates
    ]
    score, sequence_key, start = min(ranked, key=lambda item: (item[0], item[1], item[2]))
    selected = ordered_sequences[sequence_key].iloc[start : start + window_size].copy()
    selected.attrs["static_context_selection"] = {
        "sequence_id": sequence_key[-1],
        "start_row": int(start),
        "start_timestamp_s": float(selected["timestamp_s"].iloc[0]),
        "end_timestamp_s": float(selected["timestamp_s"].iloc[-1]),
        "score": float(score),
    }
    return selected


def _standardize_selected_columns(values: np.ndarray, columns: list[str], stats: dict) -> np.ndarray:
    all_columns = list(stats["columns"])
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    indices = [all_columns.index(column) for column in columns]
    return ((values - mean[indices]) / std[indices]).astype(np.float32)


def build_static_context_windows(
    df: pd.DataFrame,
    static_context: dict | None,
    feature_stats: dict,
    metadata_out: dict[str, dict] | None = None,
) -> dict[str, np.ndarray]:
    """Build normalized long static windows keyed by sensor or sequence."""
    if not static_context_enabled(static_context):
        return {}
    columns = _static_columns(static_context)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing static context columns: {missing}")
    missing_stats = [column for column in columns if column not in feature_stats["columns"]]
    if missing_stats:
        raise ValueError(f"Static context columns must be included in feature stats: {missing_stats}")
    window_size = int((static_context or {}).get("window_size", 1024))
    reference_scope = str((static_context or {}).get("reference_scope", "")).lower()
    if reference_scope == "train_global":
        source = select_low_motion_source(df, window_size)
        values = source[columns].to_numpy(dtype=np.float32)
        windows = {
            TRAIN_GLOBAL_STATIC_CONTEXT_KEY: _standardize_selected_columns(
                _fixed_length_window(values, window_size), columns, feature_stats
            )
        }
        if metadata_out is not None:
            metadata = dict(source.attrs.get("static_context_selection", {}))
            if "low_sensor" in source.columns:
                metadata["source_sensor"] = str(source["low_sensor"].iloc[0])
            metadata["reference_scope"] = reference_scope
            metadata_out[TRAIN_GLOBAL_STATIC_CONTEXT_KEY] = metadata
        return windows

    group_column = "low_sensor" if "low_sensor" in df.columns else "sequence_id"
    windows = {}
    for key, group in df.groupby(group_column, sort=False):
        if str((static_context or {}).get("selection", "")).lower() == "auto_low_motion":
            source = select_low_motion_source(group, window_size)
        else:
            source = _longest_static_source(group.sort_values("timestamp_s").reset_index(drop=True))
        values = source[columns].to_numpy(dtype=np.float32)
        window = _fixed_length_window(values, window_size)
        windows[str(key)] = _standardize_selected_columns(window, columns, feature_stats)
        if metadata_out is not None:
            metadata_out[str(key)] = dict(source.attrs.get("static_context_selection", {}))
    return windows


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
        feature_columns: list[str] | None = None,
        static_context: dict | None = None,
        bias_auxiliary: dict | None = None,
        static_windows: dict[str, np.ndarray] | None = None,
        static_window_metadata: dict[str, dict] | None = None,
        allow_static_context_build: bool = True,
        allow_train_global_static_context_build: bool = False,
    ) -> None:
        self.feature_columns = list(feature_columns or FEATURE_COLUMNS)
        if self.feature_columns[: len(LOW_IMU_COLUMNS)] != LOW_IMU_COLUMNS:
            raise ValueError(f"feature_columns must start with {LOW_IMU_COLUMNS}")
        self.df = ensure_training_columns(df, self.feature_columns)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.target_mode = target_mode
        self.competitor_columns = competitor_columns
        self.static_context = dict(static_context or {})
        self.static_context_enabled = static_context_enabled(self.static_context)
        self.bias_auxiliary = dict(bias_auxiliary or {})
        self.bias_auxiliary_enabled = bool(self.bias_auxiliary.get("enabled", False))
        self.bias_auxiliary_static_fraction = float(self.bias_auxiliary.get("stationary_fraction_threshold", 0.8))
        if self.target_mode not in {"direct", "residual", "competitor_residual"}:
            raise ValueError(f"Unsupported target_mode: {self.target_mode}")
        if self.target_mode == "competitor_residual" and self.competitor_columns is None:
            raise ValueError("target_mode='competitor_residual' requires competitor baseline columns")
        if self.competitor_columns is not None:
            missing_competitor = [col for col in self.competitor_columns if col not in self.df.columns]
            if missing_competitor:
                raise ValueError(f"Missing competitor baseline columns: {missing_competitor}")
        self.feature_stats = feature_stats or compute_standardization(self.df, self.feature_columns)
        target_values = self._target_values(self.df)
        target_frame = pd.DataFrame(target_values, columns=TARGET_COLUMNS)
        self.target_stats = target_stats or compute_standardization(target_frame, TARGET_COLUMNS)
        if target_stats is None and zero_target_mean:
            self.target_stats = {**self.target_stats, "mean": [0.0] * len(TARGET_COLUMNS)}
        self.static_window_metadata = {
            str(key): dict(value) for key, value in (static_window_metadata or {}).items()
        }
        if static_windows is not None:
            self.static_windows = {
                str(key): np.asarray(value, dtype=np.float32).copy() for key, value in static_windows.items()
            }
        elif (
            str(self.static_context.get("reference_scope", "")).lower() == "train_global"
            and not allow_train_global_static_context_build
        ):
            raise ValueError(
                "Missing precomputed train-global static context; set "
                "allow_train_global_static_context_build=True only for the training dataset"
            )
        elif self.static_context_enabled and not allow_static_context_build:
            raise ValueError("Missing precomputed static context for a dataset that may not build its own")
        else:
            self.static_windows = build_static_context_windows(
                self.df,
                self.static_context,
                self.feature_stats,
                metadata_out=self.static_window_metadata,
            )
        self.windows: list[tuple] = []
        for _, group in self.df.groupby("sequence_id", sort=False):
            group = group.sort_values("timestamp_s").reset_index(drop=True)
            static_key = (
                TRAIN_GLOBAL_STATIC_CONTEXT_KEY
                if str(self.static_context.get("reference_scope", "")).lower() == "train_global"
                else _static_group_key(group)
            )
            if self.static_context_enabled and static_key not in self.static_windows:
                raise ValueError(f"Missing precomputed static context for key: {static_key}")
            features = group[self.feature_columns].to_numpy(dtype=np.float32)
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
                    window = (features[slc], targets[slc])
                else:
                    assert physical_competitors is not None
                    window = (features[slc], targets[slc], competitors[slc], physical_competitors[slc])
                if self.static_context_enabled:
                    window = (*window, static_key)
                if self.bias_auxiliary_enabled:
                    window_frame = group.iloc[slc]
                    if "motion_label" in window_frame.columns:
                        static_fraction = float(
                            window_frame["motion_label"].astype(str).str.lower().eq("stationary").mean()
                        )
                    else:
                        static_fraction = 0.0
                    bias_mask = np.asarray(
                        [1.0 if static_fraction >= self.bias_auxiliary_static_fraction else 0.0],
                        dtype=np.float32,
                    )
                    bias_target = targets[slc].mean(axis=0, keepdims=True).astype(np.float32)
                    window = (*window, bias_target, bias_mask)
                self.windows.append(window)
        if not self.windows:
            raise ValueError(
                f"No windows created; samples={len(self.df)}, window_size={self.window_size}, stride={self.stride}"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        window = self.windows[index]
        static = None
        bias_target = None
        bias_mask = None
        core = window
        if self.bias_auxiliary_enabled:
            bias_target = torch.from_numpy(np.ascontiguousarray(window[-2].T))
            bias_mask = torch.from_numpy(np.ascontiguousarray(window[-1]))
            core = window[:-2]
        if self.static_context_enabled:
            static_key = str(core[-1])
            static = torch.from_numpy(np.ascontiguousarray(self.static_windows[static_key].T))
            core = core[:-1]
        x = torch.from_numpy(np.ascontiguousarray(core[0].T))
        y = torch.from_numpy(np.ascontiguousarray(core[1].T))
        extras = ()
        if static is not None:
            extras = (*extras, static)
        if bias_target is not None and bias_mask is not None:
            extras = (*extras, bias_target, bias_mask)
        if len(core) == 2:
            return (x, y, *extras)
        competitor = torch.from_numpy(np.ascontiguousarray(core[2].T))
        physical_competitor = torch.from_numpy(np.ascontiguousarray(core[3].T))
        return (x, y, competitor, physical_competitor, *extras)

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
