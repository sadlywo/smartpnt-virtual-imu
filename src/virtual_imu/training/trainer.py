"""Training loop for virtual IMU models."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader

from virtual_imu.config import save_config
from virtual_imu.eval.baselines import (
    add_system_id_competitor_columns,
    fit_system_id_baseline_state,
)
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
)
from virtual_imu.training.metrics_logger import MetricsLogger
from virtual_imu.utils.paths import git_commit
from virtual_imu.utils.seed import seed_everything


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
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(epochs, train, label="train")
    ax.plot(epochs, val, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(run_dir / f"loss_curve.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    train_df = pd.read_parquet(data_cfg["train_path"])
    val_df = pd.read_parquet(data_cfg["val_path"])
    window_size = int(data_cfg.get("window_size", 128))
    stride = int(data_cfg.get("stride", 32))
    target_mode = _target_mode_from_config(config)
    train_cfg = config.get("training", {})
    train_df, val_df, competitor_columns, competitor_metadata = _add_system_id_competitor(train_df, val_df, train_cfg)
    train_ds = IMUWindowDataset(
        train_df,
        window_size,
        stride,
        target_mode=target_mode,
        competitor_columns=competitor_columns,
        zero_target_mean=target_mode == "competitor_residual",
    )
    val_ds = IMUWindowDataset(
        val_df,
        window_size,
        stride,
        train_ds.feature_stats,
        train_ds.target_stats,
        target_mode=target_mode,
        competitor_columns=competitor_columns,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(data_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(val_ds, batch_size=int(data_cfg.get("batch_size", 32)), shuffle=False)

    model_cfg = dict(config.get("model", {}))
    model_cfg["in_channels"] = len(FEATURE_COLUMNS)
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

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.get("output", {}).get("runs_dir", "runs")) / f"{config.get('experiment_name', 'run')}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    if config_path:
        (run_dir / "source_config_path.txt").write_text(str(config_path), encoding="utf-8")
    (run_dir / "provenance.json").write_text(json.dumps({"git_commit": git_commit()}, indent=2), encoding="utf-8")
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

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            if len(batch) >= 3:
                x, y, competitor = batch[:3]
                competitor = competitor.to(device)
            else:
                x, y = batch
                competitor = None
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss, _ = combined_loss(pred, y, train_cfg_effective, dt=dt, competitor=competitor)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        val_inputs = []
        val_preds = []
        val_targets = []
        val_competitors = []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) >= 3:
                    x, y, competitor = batch[:3]
                    competitor_device = competitor.to(device)
                    if len(batch) >= 4:
                        val_competitors.append(batch[3].detach().cpu().numpy())
                else:
                    x, y = batch
                    competitor_device = None
                y_device = y.to(device)
                pred = model(x.to(device))
                loss, _ = combined_loss(pred, y_device, train_cfg_effective, dt=dt, competitor=competitor_device)
                val_losses.append(float(loss.detach().cpu()))
                val_inputs.append(x.detach().cpu().numpy())
                val_preds.append(pred.detach().cpu().numpy())
                val_targets.append(y.detach().cpu().numpy())

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
            "val_loss": float(np.mean(val_losses)),
            **score_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        metrics.log(row)
        monitor = float(row.get(checkpoint_metric, row["val_loss"]))
        if scheduler is not None:
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
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_cfg,
                    "config": config,
                    "feature_stats": train_ds.feature_stats,
                    "target_stats": train_ds.target_stats,
                    "feature_columns": FEATURE_COLUMNS,
                    "target_columns": TARGET_COLUMNS,
                    "target_mode": target_mode,
                    "competitor_columns": competitor_columns or [],
                    "competitor_baseline": competitor_metadata,
                    "dt": dt,
                    "git_commit": git_commit(),
                    "best_metric": best_metric,
                    "checkpoint_metric": checkpoint_metric,
                    "best_val_loss": row["val_loss"],
                    "best_val_score": row["val_score"],
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
