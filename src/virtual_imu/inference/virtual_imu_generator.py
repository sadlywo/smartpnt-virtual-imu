"""Generate virtual high-grade IMU Parquet files from checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from virtual_imu.eval.baselines import predict_system_id_from_state, select_system_id_competitor
from virtual_imu.io.parquet_store import write_parquet
from virtual_imu.models import make_model
from virtual_imu.preprocessing.normalization import apply_standardization
from virtual_imu.training.checkpointing import load_checkpoint
from virtual_imu.training.dataset import (
    FEATURE_COLUMNS,
    LOW_IMU_COLUMNS,
    TARGET_COLUMNS,
    TRAIN_GLOBAL_STATIC_CONTEXT_KEY,
    build_static_context_windows,
    ensure_training_columns,
    static_context_enabled,
)

VIRTUAL_COLUMNS = ["gx_virtual", "gy_virtual", "gz_virtual", "ax_virtual", "ay_virtual", "az_virtual"]


class VirtualIMUGenerator:
    """Sliding-window virtual IMU generator."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        self.checkpoint = load_checkpoint(self.checkpoint_path, map_location=self.device)
        self.model = make_model(self.checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()
        self.feature_stats = self.checkpoint["feature_stats"]
        self.feature_columns = list(self.checkpoint.get("feature_columns", FEATURE_COLUMNS))
        self.target_stats = self.checkpoint["target_stats"]
        self.target_mode = str(self.checkpoint.get("target_mode", "direct"))
        self.competitor_baseline = self.checkpoint.get("competitor_baseline", {})
        self.static_context = dict(self.checkpoint.get("static_context", {}))
        self.has_static_context = static_context_enabled(self.static_context)
        self.static_windows = {
            str(key): np.asarray(value, dtype=np.float32)
            for key, value in self.checkpoint.get("static_windows", {}).items()
        }
        self.config = self.checkpoint.get("config", {})
        self.window_size = int(self.config.get("data", {}).get("window_size", 128))
        self.stride = int(self.config.get("data", {}).get("stride", max(1, self.window_size // 2)))

    def _predict_competitor(self, group: pd.DataFrame) -> np.ndarray:
        """Predict the checkpoint's fitted ARX/FIR competitor for one sequence."""
        state = self.competitor_baseline.get("state")
        if not state:
            raise ValueError("Checkpoint is missing system-ID competitor state required for competitor_residual inference")
        predictions = predict_system_id_from_state(state, group)
        return select_system_id_competitor(
            predictions,
            mode=str(self.competitor_baseline.get("mode", "blend")),
            blend_weight_fir=float(self.competitor_baseline.get("blend_weight_fir", 0.5)),
        )

    def _scale_competitor_correction(self, correction: np.ndarray) -> np.ndarray:
        """Apply configured gyro/accel correction scales for competitor residual mode."""
        training_cfg = self.config.get("training", {})
        scaled = correction.copy()
        scaled[:, :3] *= float(training_cfg.get("competitor_correction_scale_gyro", 1.0))
        scaled[:, 3:] *= float(training_cfg.get("competitor_correction_scale_accel", 1.0))
        return scaled

    def _predict_sequence(self, group: pd.DataFrame, static_windows: dict[str, np.ndarray] | None = None) -> np.ndarray:
        group = ensure_training_columns_for_inference(group, self.feature_columns)
        features = group[self.feature_columns].to_numpy(dtype=np.float32)
        features = apply_standardization(features, self.feature_stats).astype(np.float32)
        n = len(group)
        competitor = self._predict_competitor(group) if self.target_mode == "competitor_residual" else None
        static_tensor = None
        if self.has_static_context:
            reference_scope = str(self.static_context.get("reference_scope", "")).lower()
            key = (
                TRAIN_GLOBAL_STATIC_CONTEXT_KEY
                if reference_scope == "train_global"
                else str(group["low_sensor"].iloc[0]) if "low_sensor" in group.columns else str(group["sequence_id"].iloc[0])
            )
            if not static_windows or key not in static_windows:
                if reference_scope in {"train_sensor", "train_global"}:
                    raise ValueError(f"Checkpoint is missing train-derived static context for sensor key: {key}")
                static_windows = build_static_context_windows(group, self.static_context, self.feature_stats)
            static_tensor = torch.from_numpy(static_windows[key].T[None, ...]).to(self.device)
        pred_sum = np.zeros((n, len(TARGET_COLUMNS)), dtype=np.float64)
        pred_count = np.zeros((n, 1), dtype=np.float64)
        starts = list(range(0, max(n - self.window_size + 1, 1), self.stride))
        if starts[-1] != max(n - self.window_size, 0):
            starts.append(max(n - self.window_size, 0))
        with torch.no_grad():
            for start in starts:
                end = min(start + self.window_size, n)
                window = features[start:end]
                pad = self.window_size - len(window)
                if pad > 0:
                    window = np.pad(window, ((0, pad), (0, 0)), mode="edge")
                tensor = torch.from_numpy(window.T[None, ...]).to(self.device)
                if static_tensor is None:
                    pred = self.model(tensor).detach().cpu().numpy()[0].T
                else:
                    pred = self.model(tensor, static_tensor).detach().cpu().numpy()[0].T
                pred = pred[: end - start]
                pred = apply_standardization(pred, self.target_stats, inverse=True)
                if self.target_mode == "residual":
                    pred = group.iloc[start:end][LOW_IMU_COLUMNS].to_numpy(dtype=float) + pred
                elif self.target_mode == "competitor_residual":
                    assert competitor is not None
                    pred = competitor[start:end] + self._scale_competitor_correction(pred)
                pred_sum[start:end] += pred
                pred_count[start:end] += 1.0
        return pred_sum / np.maximum(pred_count, 1.0)

    def run(self, input_path: str | Path, out_path: str | Path) -> Path:
        """Run inference on a Parquet file and write virtual IMU predictions."""
        logger.info("Reading virtual IMU inference input from {}", input_path)
        df = pd.read_parquet(input_path)
        self.run_frame(df, out_path)
        return Path(out_path)

    def run_frame(self, df: pd.DataFrame, out_path: str | Path | None = None) -> pd.DataFrame:
        """Run inference directly on an in-memory frame, optionally writing Parquet."""
        if "sequence_id" not in df.columns:
            df = df.copy()
            df["sequence_id"] = "sequence"
        df = ensure_training_columns_for_inference(df, self.feature_columns)
        static_windows: dict[str, np.ndarray] | None = None
        if self.has_static_context:
            if self.static_windows:
                static_windows = self.static_windows
            elif str(self.static_context.get("reference_scope", "")).lower() not in {"train_sensor", "train_global"}:
                static_windows = build_static_context_windows(df, self.static_context, self.feature_stats)
        outputs: list[pd.DataFrame] = []
        for _, group in df.groupby("sequence_id", sort=False):
            group = group.sort_values("timestamp_s").reset_index(drop=True)
            pred = self._predict_sequence(group, static_windows)
            out = group.copy()
            for idx, col in enumerate(VIRTUAL_COLUMNS):
                out[col] = pred[:, idx]
            outputs.append(out)
        result = pd.concat(outputs, ignore_index=True)
        ordered = _output_columns(result)
        result = result[ordered]
        if out_path is not None:
            path = write_parquet(result, out_path)
            logger.info("Wrote virtual IMU predictions to {}", path)
        return result


def ensure_training_columns_for_inference(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    """Ensure feature columns exist without requiring high-grade targets."""
    out = df.copy()
    if "temperature_c" not in out:
        out["temperature_c"] = 25.0
    features = list(feature_columns or FEATURE_COLUMNS)
    missing = [col for col in features if col not in out.columns]
    if missing:
        raise ValueError(f"Missing required low-grade columns for inference: {missing}")
    for target in TARGET_COLUMNS:
        if target not in out:
            out[target] = 0.0
    return ensure_training_columns(out, features)


def _output_columns(df: pd.DataFrame) -> list[str]:
    base = [
        "timestamp_s",
        "gx_low",
        "gy_low",
        "gz_low",
        "ax_low",
        "ay_low",
        "az_low",
        *VIRTUAL_COLUMNS,
    ]
    high = [col for col in TARGET_COLUMNS if col in df.columns]
    extra = [
        col
        for col in [
            "temperature_c", "motion_label", "sequence_id", "original_sequence_id",
            "dataset_id", "low_sensor", "high_sensor", "pair_id", "reference_frame",
            "source_timestamp_s", "available_timestamp_s",
        ]
        if col in df.columns
    ]
    return [col for col in [*base, *high, *extra] if col in df.columns]
