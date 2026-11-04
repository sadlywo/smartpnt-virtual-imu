"""Training loop for virtual IMU models."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader

from virtual_imu.config import save_config
from virtual_imu.eval.baselines import (
    add_system_id_competitor_columns,
    fit_system_id_baseline_state,
)
from virtual_imu.io.training_split_manifest import TrainingSplitBundle, load_training_split_manifest
from virtual_imu.models import make_model
from virtual_imu.models.losses import combined_loss
from virtual_imu.preprocessing.normalization import apply_standardization
from virtual_imu.training.checkpointing import save_checkpoint
from virtual_imu.training.dataset import (
    COMPETITOR_COLUMNS,
    FEATURE_COLUMNS,
    LOW_IMU_COLUMNS,
    TARGET_COLUMNS,
    IMUWindowDataset,
    static_context_enabled,
)
from virtual_imu.training.metrics_logger import MetricsLogger
from virtual_imu.training.paired_validation import grouped_relative_rmse_validation
from virtual_imu.training.schedules import warmup_cosine_scheduler
from virtual_imu.training.unified_data import (
    dataset_balanced_sampler,
    snapshot_unified_training_inputs,
)
from virtual_imu.utils.paths import git_commit
from virtual_imu.utils.seed import seed_everything
from virtual_imu.viz.style import new_figure, save_figure


def _device_from_config(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _estimate_dt(df: pd.DataFrame) -> float:
    diffs = df.sort_values("timestamp_s")["timestamp_s"].diff().dropna().to_numpy()
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(np.median(diffs)) if diffs.size else 0.01


def _plot_loss_curve(history: list[dict[str, float]], run_dir: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    train = [row["train_loss"] for row in history]
    val = [row["val_loss"] for row in history]
    fig, ax = new_figure("single")
    ax.plot(epochs, train, label="train")
    ax.plot(epochs, val, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, run_dir, "loss_curve")


def _target_mode_from_config(config: dict[str, Any]) -> str:
    return str(
        config.get("model", {}).get(
            "target_mode",
            config.get("data", {}).get("target_mode", config.get("training", {}).get("target_mode", "direct")),
        )
    )


def _competitor_enabled(train_cfg: dict[str, Any]) -> bool:
    competitor_cfg = train_cfg.get("competitor_baseline", {})
    validation_cfg = train_cfg.get("validation_score", {})
    return bool(
        competitor_cfg.get("enabled", False)
        or float(train_cfg.get("baseline_margin_weight", 0.0)) > 0.0
        or bool(validation_cfg.get("relative_to_competitor", False))
    )


def _feature_columns_from_config(data_cfg: dict[str, Any]) -> list[str]:
    columns = list(data_cfg.get("feature_columns", FEATURE_COLUMNS))
    if columns[: len(LOW_IMU_COLUMNS)] != LOW_IMU_COLUMNS:
        raise ValueError(f"data.feature_columns must start with {LOW_IMU_COLUMNS}")
    return columns


def _load_training_frames(
    data_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, TrainingSplitBundle | None]:
    split_manifest = data_cfg.get("split_manifest")
    strict_manifest = data_cfg.get("strict_manifest")
    if split_manifest or strict_manifest:
        if not split_manifest or not strict_manifest:
            raise ValueError("data.split_manifest and data.strict_manifest must be configured together")
        options = {"split_names": data_cfg["load_splits"]} if data_cfg.get("load_splits") else {}
        bundle = load_training_split_manifest(Path(split_manifest), Path(strict_manifest), **options)
        return bundle.frames["train"], bundle.frames["validation"], bundle
    logger.info("READS legacy training Parquet {}", data_cfg["train_path"])
    logger.info("READS legacy validation Parquet {}", data_cfg["val_path"])
    return pd.read_parquet(data_cfg["train_path"]), pd.read_parquet(data_cfg["val_path"]), None


def _validate_finite_features(df: pd.DataFrame, feature_columns: list[str], split_name: str) -> None:
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{split_name} split is missing configured feature columns: {missing}")
    values = df[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{split_name} split contains non-finite configured feature values")


def _forward_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    static_context: torch.Tensor | None = None,
) -> torch.Tensor:
    if static_context is None:
        return model(x)
    return model(x, static_context)


def _unpack_batch(
    batch: tuple[torch.Tensor, ...],
    has_static_context: bool,
    has_bias_auxiliary: bool,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    np.ndarray | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    static_tensor = None
    bias_target = None
    bias_mask = None
    core = batch
    if has_bias_auxiliary:
        bias_target = batch[-2].to(device)
        bias_mask = batch[-1].to(device)
        core = batch[:-2]
    if has_static_context:
        static_tensor = core[-1].to(device)
        core = core[:-1]
    if len(core) >= 3:
        x, y, competitor = core[:3]
        competitor_device = competitor.to(device)
    else:
        x, y = core
        competitor_device = None
    physical_competitor = core[3].detach().cpu().numpy() if len(core) >= 4 else None
    return x.to(device), y.to(device), competitor_device, static_tensor, physical_competitor, bias_target, bias_mask


def _bias_auxiliary_loss(
    model: torch.nn.Module,
    bias_target: torch.Tensor | None,
    bias_mask: torch.Tensor | None,
    train_cfg: dict[str, Any],
) -> torch.Tensor | None:
    cfg = train_cfg.get("bias_auxiliary", {})
    weight = float(cfg.get("weight", 0.0))
    if weight <= 0.0 or bias_target is None or bias_mask is None:
        return None
    bias_offset = getattr(model, "last_aux", {}).get("bias_offset")
    if bias_offset is None:
        return None
    per_sample = F.smooth_l1_loss(bias_offset, bias_target, reduction="none").mean(dim=(1, 2))
    mask = bias_mask.reshape(-1).to(dtype=per_sample.dtype)
    denom = torch.clamp(mask.sum(), min=1.0)
    return weight * (per_sample * mask).sum() / denom


def _add_system_id_competitor(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str] | None, dict[str, Any]]:
    """Attach ARX/FIR competitor columns when margin training is enabled."""
    if not _competitor_enabled(train_cfg):
        return train_df, val_df, None, {}

    competitor_cfg = train_cfg.get("competitor_baseline", {})
    metadata = {
        "enabled": True,
        "mode": str(competitor_cfg.get("mode", "blend")),
        "fir_taps": int(competitor_cfg.get("fir_taps", 9)),
        "blend_weight_fir": float(competitor_cfg.get("blend_weight_fir", 0.5)),
        "ridge_alpha": float(competitor_cfg.get("ridge_alpha", 1e-2)),
        "columns": COMPETITOR_COLUMNS,
    }
    logger.info(
        "Adding {} ARX/FIR competitor baseline columns with fir_taps={}",
        metadata["mode"],
        metadata["fir_taps"],
    )
    state = fit_system_id_baseline_state(
        train_df,
        fir_taps=metadata["fir_taps"],
        alpha=metadata["ridge_alpha"],
    )
    metadata["state"] = state
    train_with_competitor = add_system_id_competitor_columns(
        train_df,
        train_df,
        fir_taps=metadata["fir_taps"],
        mode=metadata["mode"],
        blend_weight_fir=metadata["blend_weight_fir"],
        alpha=metadata["ridge_alpha"],
        state=state,
    )
    val_with_competitor = add_system_id_competitor_columns(
        train_df,
        val_df,
        fir_taps=metadata["fir_taps"],
        mode=metadata["mode"],
        blend_weight_fir=metadata["blend_weight_fir"],
        alpha=metadata["ridge_alpha"],
        state=state,
    )
    return train_with_competitor, val_with_competitor, COMPETITOR_COLUMNS, metadata


def _inverse_channel_first(values: np.ndarray, stats: dict) -> np.ndarray:
    flat = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
    return apply_standardization(flat, stats, inverse=True)


def _axis_coherence_proxy(x: np.ndarray, y: np.ndarray, nperseg: int = 128) -> float:
    """Estimate magnitude-squared coherence with overlapped FFT segments."""
    if len(x) < 16 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    nperseg = min(nperseg, max(16, len(x) // 2))
    step = max(1, nperseg // 2)
    starts = list(range(0, len(x) - nperseg + 1, step))
    if len(starts) < 2:
        return float("nan")
    window = np.hanning(nperseg)
    pxx = None
    pyy = None
    pxy = None
    for start in starts:
        xs = (x[start : start + nperseg] - np.mean(x[start : start + nperseg])) * window
        ys = (y[start : start + nperseg] - np.mean(y[start : start + nperseg])) * window
        xf = np.fft.rfft(xs)
        yf = np.fft.rfft(ys)
        pxx = np.abs(xf) ** 2 if pxx is None else pxx + np.abs(xf) ** 2
        pyy = np.abs(yf) ** 2 if pyy is None else pyy + np.abs(yf) ** 2
        pxy = xf * np.conj(yf) if pxy is None else pxy + xf * np.conj(yf)
    assert pxx is not None and pyy is not None and pxy is not None
    pxx = pxx / len(starts)
    pyy = pyy / len(starts)
    pxy = pxy / len(starts)
    coh = (np.abs(pxy) ** 2) / (pxx * pyy + 1e-12)
    finite = coh[1:][np.isfinite(coh[1:])]
    return float(np.mean(finite)) if finite.size else float("nan")


def _gyro_coherence_proxy(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute a compact segmented coherence proxy over gyro axes."""
    coherences = []
    for axis in range(3):
        coherence = _axis_coherence_proxy(pred[:, axis], target[:, axis])
        if np.isfinite(coherence):
            coherences.append(coherence)
    return float(np.mean(coherences)) if coherences else 0.0


def _apply_competitor_correction_scale(correction: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """Scale competitor-residual corrections before physical validation."""
    gyro_scale = float(cfg.get("competitor_correction_scale_gyro", 1.0))
    accel_scale = float(cfg.get("competitor_correction_scale_accel", 1.0))
    scaled = correction.copy()
    scaled[:, :3] *= gyro_scale
    scaled[:, 3:] *= accel_scale
    return scaled


def _validation_composite_score(
    input_batches: list[np.ndarray],
    pred_batches: list[np.ndarray],
    target_batches: list[np.ndarray],
    competitor_physical_batches: list[np.ndarray],
    feature_stats: dict,
    target_stats: dict,
    target_mode: str,
    dt: float,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Compute a physical-scale validation score from prediction/target windows."""
    if not input_batches or not pred_batches or not target_batches:
        return {
            "val_score": float("inf"),
            "val_gyro_rmse": float("inf"),
            "val_accel_rmse": float("inf"),
            "val_gyro_spectral": float("inf"),
            "val_gyro_coherence": 0.0,
            "val_gyro_coherence_penalty": float("inf"),
            "val_preintegration": float("inf"),
            "val_competitor_gyro_rmse": float("nan"),
            "val_competitor_accel_rmse": float("nan"),
            "val_competitor_gyro_coherence": float("nan"),
            "val_competitor_preintegration": float("nan"),
            "val_relative_gyro_rmse": float("nan"),
            "val_relative_accel_rmse": float("nan"),
            "val_relative_preintegration": float("nan"),
            "val_competitor_coherence_gap": float("nan"),
        }
    inputs = _inverse_channel_first(np.concatenate(input_batches, axis=0), feature_stats)
    pred_target = _inverse_channel_first(np.concatenate(pred_batches, axis=0), target_stats)
    target = _inverse_channel_first(np.concatenate(target_batches, axis=0), target_stats)
    low = inputs[:, : len(LOW_IMU_COLUMNS)]
    competitor = None
    if competitor_physical_batches:
        competitor = np.concatenate(competitor_physical_batches, axis=0).transpose(0, 2, 1).reshape(-1, len(TARGET_COLUMNS))
    if target_mode == "residual":
        pred = low + pred_target
        target = low + target
    elif target_mode == "competitor_residual":
        if competitor is None:
            raise ValueError("competitor_residual validation requires physical competitor batches")
        pred = competitor + _apply_competitor_correction_scale(pred_target, cfg)
        target = competitor + target
    else:
        pred = pred_target
    err = pred - target
    gyro_rmse = float(np.sqrt(np.mean(np.sum(err[:, :3] ** 2, axis=1))))
    accel_rmse = float(np.sqrt(np.mean(np.sum(err[:, 3:] ** 2, axis=1))))
    pred_fft = np.abs(np.fft.rfft(pred[:, :3], axis=0))
    target_fft = np.abs(np.fft.rfft(target[:, :3], axis=0))
    gyro_spectral = float(np.mean(np.abs(np.log1p(pred_fft) - np.log1p(target_fft))))
    gyro_coherence = _gyro_coherence_proxy(pred[:, :3], target[:, :3])
    gyro_coherence_penalty = 1.0 - gyro_coherence
    increments = np.cumsum(err, axis=0) * dt
    preintegration = float(np.mean(np.linalg.norm(increments, axis=1)))
    weights = cfg.get("validation_score", {})
    absolute_score = (
        float(weights.get("gyro_rmse_weight", 1.0)) * gyro_rmse
        + float(weights.get("accel_rmse_weight", 1.0)) * accel_rmse
        + float(weights.get("gyro_spectral_weight", 0.5)) * gyro_spectral
        + float(weights.get("gyro_coherence_weight", 0.0)) * gyro_coherence_penalty
        + float(weights.get("preintegration_weight", 0.5)) * preintegration
    )
    score = absolute_score
    competitor_gyro_rmse = float("nan")
    competitor_accel_rmse = float("nan")
    competitor_gyro_coherence = float("nan")
    competitor_preintegration = float("nan")
    relative_gyro_rmse = float("nan")
    relative_accel_rmse = float("nan")
    relative_preintegration = float("nan")
    competitor_coherence_gap = float("nan")
    if competitor is not None:
        comp_err = competitor - target
        competitor_gyro_rmse = float(np.sqrt(np.mean(np.sum(comp_err[:, :3] ** 2, axis=1))))
        competitor_accel_rmse = float(np.sqrt(np.mean(np.sum(comp_err[:, 3:] ** 2, axis=1))))
        competitor_gyro_coherence = _gyro_coherence_proxy(competitor[:, :3], target[:, :3])
        comp_increments = np.cumsum(comp_err, axis=0) * dt
        competitor_preintegration = float(np.mean(np.linalg.norm(comp_increments, axis=1)))
        eps = 1e-12
        relative_gyro_rmse = gyro_rmse / max(competitor_gyro_rmse, eps)
        relative_accel_rmse = accel_rmse / max(competitor_accel_rmse, eps)
        relative_preintegration = preintegration / max(competitor_preintegration, eps)
        competitor_coherence_gap = max(0.0, competitor_gyro_coherence - gyro_coherence)
        if bool(weights.get("relative_to_competitor", False)):
            score += (
                float(weights.get("relative_gyro_rmse_weight", 2.0)) * relative_gyro_rmse
                + float(weights.get("relative_accel_rmse_weight", 0.8)) * relative_accel_rmse
                + float(weights.get("relative_preintegration_weight", 1.0)) * relative_preintegration
                + float(weights.get("competitor_coherence_gap_weight", 1.5)) * competitor_coherence_gap
            )
    return {
        "val_score": float(score),
        "val_gyro_rmse": gyro_rmse,
        "val_accel_rmse": accel_rmse,
        "val_gyro_spectral": gyro_spectral,
        "val_gyro_coherence": gyro_coherence,
        "val_gyro_coherence_penalty": gyro_coherence_penalty,
        "val_preintegration": preintegration,
        "val_competitor_gyro_rmse": competitor_gyro_rmse,
        "val_competitor_accel_rmse": competitor_accel_rmse,
        "val_competitor_gyro_coherence": competitor_gyro_coherence,
        "val_competitor_preintegration": competitor_preintegration,
        "val_relative_gyro_rmse": relative_gyro_rmse,
        "val_relative_accel_rmse": relative_accel_rmse,
        "val_relative_preintegration": relative_preintegration,
        "val_competitor_coherence_gap": competitor_coherence_gap,
    }


def train_model(config: dict[str, Any], config_path: str | Path | None = None) -> Path:
    """Train a model from YAML-like config and return the run directory."""
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    data_cfg = config["data"]
    train_df, val_df, split_bundle = _load_training_frames(data_cfg)
    minimum_samples = int(data_cfg.get("minimum_sequence_samples", 0))
    if minimum_samples:
        for split_name, frame in (("train", train_df), ("validation", val_df)):
            sizes = frame.groupby("sequence_id")["sequence_id"].transform("size")
            if bool((sizes < minimum_samples).any()):
                raise ValueError(f"{split_name} contains a sequence below the fixed common-context support {minimum_samples}")
    window_size = int(data_cfg.get("window_size", 128))
    stride = int(data_cfg.get("stride", 32))
    feature_columns = _feature_columns_from_config(data_cfg)
    _validate_finite_features(train_df, feature_columns, "train")
    _validate_finite_features(val_df, feature_columns, "validation")
    static_context_cfg = dict(data_cfg.get("static_context", {}))
    has_static_context = static_context_enabled(static_context_cfg)
    train_derived_static_context = str(static_context_cfg.get("reference_scope", "")).lower() in {
        "train_sensor",
        "train_global",
    }
    target_mode = _target_mode_from_config(config)
    train_cfg = config.get("training", {})
    grouped_validation = train_cfg.get("validation_score", {}).get("method") == "grouped_relative_rmse"
    if grouped_validation and target_mode not in {"direct", "residual"}:
        raise ValueError("grouped_relative_rmse validation supports only direct and residual target modes")
    bias_auxiliary_cfg = dict(train_cfg.get("bias_auxiliary", {}))
    has_bias_auxiliary = bool(bias_auxiliary_cfg.get("enabled", False))
    train_df, val_df, competitor_columns, competitor_metadata = _add_system_id_competitor(train_df, val_df, train_cfg)
    train_ds = IMUWindowDataset(
        train_df,
        window_size,
        stride,
        target_mode=target_mode,
        competitor_columns=competitor_columns,
        zero_target_mean=target_mode == "competitor_residual",
        feature_columns=feature_columns,
        static_context=static_context_cfg,
        bias_auxiliary=bias_auxiliary_cfg,
        allow_train_global_static_context_build=(
            str(static_context_cfg.get("reference_scope", "")).lower() == "train_global"
        ),
    )
    val_ds = IMUWindowDataset(
        val_df,
        window_size,
        stride,
        train_ds.feature_stats,
        train_ds.target_stats,
        target_mode=target_mode,
        competitor_columns=competitor_columns,
        feature_columns=feature_columns,
        static_context=static_context_cfg,
        bias_auxiliary=bias_auxiliary_cfg,
        static_windows=(
            train_ds.static_windows
            if train_derived_static_context
            else None
        ),
        static_window_metadata=(
            train_ds.static_window_metadata
            if train_derived_static_context
            else None
        ),
        allow_static_context_build=not train_derived_static_context,
    )
    sampler = (
        dataset_balanced_sampler(train_df, window_size=window_size, stride=stride, seed=seed)
        if data_cfg.get("sampling") == "dataset_balanced" else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(val_ds, batch_size=int(data_cfg.get("batch_size", 32)), shuffle=False)

    model_cfg = dict(config.get("model", {}))
    model_cfg["in_channels"] = len(feature_columns)
    model_cfg["out_channels"] = len(TARGET_COLUMNS)
    device = _device_from_config(str(config.get("training", {}).get("device", "auto")))
    model = make_model(model_cfg).to(device)
    train_cfg_effective = {**train_cfg, "target_mode": target_mode}
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = None
    if str(train_cfg.get("lr_scheduler", "none")).lower() == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(train_cfg.get("lr_factor", 0.5)),
            patience=int(train_cfg.get("lr_patience", 4)),
            min_lr=float(train_cfg.get("min_lr", 0.0)),
        )
    elif str(train_cfg.get("lr_scheduler", "none")).lower() == "cosine_warmup":
        scheduler = warmup_cosine_scheduler(optimizer, train_cfg)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.get("output", {}).get("runs_dir", "runs")) / f"{config.get('experiment_name', 'run')}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    if config_path:
        (run_dir / "source_config_path.txt").write_text(str(config_path), encoding="utf-8")
    split_provenance: dict[str, list[str]] = {}
    provenance: dict[str, Any] = {"git_commit": git_commit(), **snapshot_unified_training_inputs(data_cfg, run_dir)}
    if split_bundle is not None:
        split_provenance = {
            split_name: [str(record["sequence_id"]) for record in split_bundle.split_manifest["splits"][split_name]]
            for split_name in ("train", "validation", "test")
        }
        provenance.update(
            {
                "split_manifest": str(split_bundle.split_manifest_path),
                "strict_manifest": str(split_bundle.strict_manifest_path),
                "split_provenance": split_provenance,
                "input_records": list(split_bundle.input_records),
            }
        )
        shutil.copy2(split_bundle.split_manifest_path, run_dir / "split_manifest.json")
        shutil.copy2(split_bundle.strict_manifest_path, run_dir / "strict_manifest.json")
        logger.info("WRITES manifest snapshots to {}", run_dir)
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    metrics = MetricsLogger(run_dir / "metrics.csv")

    dt = _estimate_dt(train_df)
    best_metric = float("inf")
    patience = int(train_cfg.get("early_stopping_patience", 5))
    checkpoint_metric = str(train_cfg.get("checkpoint_metric", "val_score"))
    stale_epochs = 0
    history: list[dict[str, float]] = []
    epochs = int(train_cfg.get("epochs", 5))
    grad_clip = float(train_cfg.get("grad_clip_norm", 0.0))
    logger.info("Training on {} with {} train windows and {} val windows", device, len(train_ds), len(val_ds))
    if grouped_validation:
        logger.info("READS original SI validation rows and unshuffled prediction windows for grouped_relative_rmse")
        logger.info("SKIPS legacy pooled-window FFT, coherence, and cumulative-integration validation diagnostics")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        train_bias_auxiliary_losses = []
        for batch in train_loader:
            x, y, competitor, static_tensor, _, bias_target, bias_mask = _unpack_batch(
                batch,
                has_static_context,
                has_bias_auxiliary,
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            pred = _forward_model(model, x, static_tensor)
            loss, _ = combined_loss(pred, y, train_cfg_effective, dt=dt, competitor=competitor)
            bias_aux = _bias_auxiliary_loss(model, bias_target, bias_mask, train_cfg_effective)
            if bias_aux is not None:
                loss = loss + bias_aux
                train_bias_auxiliary_losses.append(float(bias_aux.detach().cpu()))
            else:
                train_bias_auxiliary_losses.append(0.0)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss during optimization")
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        val_bias_auxiliary_losses = []
        val_inputs = []
        val_preds = []
        val_targets = []
        val_competitors = []
        with torch.no_grad():
            for batch in val_loader:
                x_device, y_device, competitor_device, static_tensor, physical_competitor, bias_target, bias_mask = _unpack_batch(
                    batch,
                    has_static_context,
                    has_bias_auxiliary,
                    device,
                )
                if physical_competitor is not None and not grouped_validation:
                    val_competitors.append(physical_competitor)
                pred = _forward_model(model, x_device, static_tensor)
                loss, _ = combined_loss(pred, y_device, train_cfg_effective, dt=dt, competitor=competitor_device)
                bias_aux = _bias_auxiliary_loss(model, bias_target, bias_mask, train_cfg_effective)
                if bias_aux is not None:
                    loss = loss + bias_aux
                    val_bias_auxiliary_losses.append(float(bias_aux.detach().cpu()))
                else:
                    val_bias_auxiliary_losses.append(0.0)
                val_losses.append(float(loss.detach().cpu()))
                val_preds.append(pred.detach().cpu().numpy())
                if not grouped_validation:
                    val_inputs.append(x_device.detach().cpu().numpy())
                    val_targets.append(y_device.detach().cpu().numpy())

        if grouped_validation:
            score_metrics, score_details = grouped_relative_rmse_validation(
                val_df,
                val_preds,
                train_ds.target_stats,
                window_size=window_size,
                stride=stride,
                target_mode=target_mode,
            )
        else:
            score_metrics = _validation_composite_score(
                val_inputs,
                val_preds,
                val_targets,
                val_competitors,
                train_ds.feature_stats,
                train_ds.target_stats,
                target_mode,
                dt,
                train_cfg,
            )
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)),
            "train_bias_auxiliary_loss": float(np.mean(train_bias_auxiliary_losses)),
            "val_loss": float(np.mean(val_losses)),
            "val_bias_auxiliary_loss": float(np.mean(val_bias_auxiliary_losses)),
            **score_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if grouped_validation:
            if checkpoint_metric not in row:
                raise ValueError(f"checkpoint_metric {checkpoint_metric!r} is not available in grouped validation metrics")
            monitor = float(row[checkpoint_metric])
            if not np.isfinite(monitor):
                raise ValueError(f"checkpoint_metric {checkpoint_metric!r} is non-finite")
            details_path = run_dir / "validation" / f"epoch_{epoch:04d}.json"
            details_path.parent.mkdir(parents=True, exist_ok=True)
            details_path.write_text(
                json.dumps(
                    {**score_details, "epoch": epoch, "checkpoint_metric": checkpoint_metric, "checkpoint_value": monitor},
                    indent=2,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            logger.info("WRITES grouped validation details {}", details_path)
        else:
            monitor = float(row.get(checkpoint_metric, row["val_loss"]))
        if not np.isfinite(monitor):
            raise FloatingPointError("Nonfinite validation selection metric during optimization")
        history.append(row)
        metrics.log(row)
        if scheduler is not None:
            if str(train_cfg.get("lr_scheduler", "none")).lower() == "cosine_warmup":
                scheduler.step()
            else:
                scheduler.step(monitor)
        logger.info(
            "Epoch {:03d}: train={:.5f} val={:.5f} score={:.5f} lr={:.2e}",
            epoch,
            row["train_loss"],
            row["val_loss"],
            row["val_score"],
            optimizer.param_groups[0]["lr"],
        )

        if monitor < best_metric:
            best_metric = monitor
            stale_epochs = 0
            validation_metadata = {}
            if grouped_validation:
                best_details_path = run_dir / "validation" / "best.json"
                shutil.copy2(details_path, best_details_path)
                logger.info("WRITES best grouped validation details {}", best_details_path)
                validation_metadata = {
                    "validation_details_path": "validation/best.json",
                    "best_val_grouped_score": row["val_grouped_score"],
                }
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_cfg,
                    "config": config,
                    "feature_stats": train_ds.feature_stats,
                    "target_stats": train_ds.target_stats,
                    "feature_columns": feature_columns,
                    "static_context": static_context_cfg,
                    "static_windows": train_ds.static_windows,
                    "static_window_metadata": train_ds.static_window_metadata,
                    "split_provenance": split_provenance,
                    "bias_auxiliary": bias_auxiliary_cfg,
                    "target_columns": TARGET_COLUMNS,
                    "target_mode": target_mode,
                    "competitor_columns": competitor_columns or [],
                    "competitor_baseline": competitor_metadata,
                    "dt": dt,
                    "git_commit": git_commit(),
                    "best_metric": best_metric,
                    "best_epoch": epoch,
                    "checkpoint_metric": checkpoint_metric,
                    "best_val_loss": row["val_loss"],
                    "best_val_score": row["val_score"],
                    **validation_metadata,
                },
                run_dir / "best.ckpt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info("Early stopping after {} stale epochs", stale_epochs)
                break

    _plot_loss_curve(history, run_dir)
    logger.info("Finished training. Best checkpoint: {}", run_dir / "best.ckpt")
    return run_dir
