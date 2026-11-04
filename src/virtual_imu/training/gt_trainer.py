"""Training loop for GT-supervised inertial odometry tasks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger

from virtual_imu.config import save_config
from virtual_imu.models import make_model
from virtual_imu.preprocessing.normalization import apply_standardization
from virtual_imu.training.checkpointing import save_checkpoint
from virtual_imu.training.gt_supervised_dataset import (
    SMARTPNT_GT_IMU_COLUMNS,
    make_smartpnt_gt_split_dataloaders,
)
from virtual_imu.training.metrics_logger import MetricsLogger
from virtual_imu.training.schedules import warmup_cosine_scheduler
from virtual_imu.training.unified_data import (
    snapshot_unified_training_inputs,
    unified_navigation_frame,
)
from virtual_imu.utils.paths import git_commit
from virtual_imu.utils.seed import seed_everything


def _device_from_config(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _skipped_sequence_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [str(key) for key in value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _loss_fn(name: str) -> torch.nn.Module:
    loss_name = name.lower()
    if loss_name in {"huber", "smooth_l1", "smoothl1"}:
        return torch.nn.SmoothL1Loss()
    if loss_name in {"mse", "l2"}:
        return torch.nn.MSELoss()
    if loss_name in {"mae", "l1"}:
        return torch.nn.L1Loss()
    raise ValueError(f"Unsupported GT-supervised loss: {name}")


def _project_prediction(pred: torch.Tensor, target_layout: str, prediction_head: str) -> torch.Tensor:
    if target_layout == "sequence":
        return pred
    head = prediction_head.lower()
    if head == "last":
        return pred[:, :, -1]
    if head == "center":
        return pred[:, :, pred.shape[-1] // 2]
    if head == "mean":
        return pred.mean(dim=-1)
    raise ValueError(f"Unsupported vector prediction_head: {prediction_head}")


def _inverse_targets(values: np.ndarray, stats: dict[str, list[float]] | None, target_layout: str) -> np.ndarray:
    if target_layout == "sequence":
        flat = values.transpose(0, 2, 1).reshape(-1, values.shape[1])
    else:
        flat = values.reshape(-1, values.shape[-1])
    return apply_standardization(flat, stats, inverse=True) if stats is not None else flat


def _inverse_targets_torch(values: torch.Tensor, stats: dict[str, list[float]] | None) -> torch.Tensor:
    if stats is None:
        return values
    mean = torch.as_tensor(stats["mean"], dtype=values.dtype, device=values.device)
    std = torch.as_tensor(stats["std"], dtype=values.dtype, device=values.device)
    return values * std + mean


def _component_rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err**2))) if err.size else float("nan")


def _vector_rmse(err: np.ndarray, indices: list[int]) -> float:
    if not indices:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(err[:, indices] ** 2, axis=1))))


def _metric_groups(target_columns: list[str]) -> dict[str, list[int]]:
    groups = {
        "position": [],
        "velocity": [],
        "attitude": [],
    }
    for idx, column in enumerate(target_columns):
        if column in {
            "gt_x_local_m",
            "gt_y_local_m",
            "gt_z_local_m",
            "delta_x_local_m",
            "delta_y_local_m",
            "delta_z_local_m",
            "delta_forward_m",
            "delta_right_m",
            "delta_up_m",
        }:
            groups["position"].append(idx)
        elif column in {"gt_v_east_mps", "gt_v_north_mps", "gt_v_up_mps"}:
            groups["velocity"].append(idx)
        elif column in {"gt_heading_deg", "gt_pitch_deg", "gt_roll_deg", "delta_heading_deg", "delta_pitch_deg", "delta_roll_deg"}:
            groups["attitude"].append(idx)
    return groups


def _meta_tensor(meta: dict[str, Any], key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    value = meta[key]
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


def _indices(target_columns: list[str], names: list[str]) -> list[int]:
    return [target_columns.index(name) for name in names if name in target_columns]


def _integration_loss(
    pred: torch.Tensor,
    meta: dict[str, Any],
    target_stats: dict[str, list[float]] | None,
    target_columns: list[str],
    criterion: torch.nn.Module,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zero = pred.new_tensor(0.0)
    velocity_weight = float(cfg.get("velocity_integration_weight", 0.0))
    heading_weight = float(cfg.get("heading_integration_weight", 0.0))
    if velocity_weight <= 0.0 and heading_weight <= 0.0:
        return zero, zero, zero
    pred_physical = _inverse_targets_torch(pred, target_stats)
    velocity_loss = zero
    heading_loss = zero
    body_indices = _indices(target_columns, ["delta_forward_m", "delta_right_m", "delta_up_m"])
    global_indices = _indices(target_columns, ["delta_x_local_m", "delta_y_local_m", "delta_z_local_m"])
    if velocity_weight > 0.0 and len(body_indices) == 3:
        aux = torch.stack(
            [
                _meta_tensor(meta, "integrated_delta_forward_m", device, pred.dtype),
                _meta_tensor(meta, "integrated_delta_right_m", device, pred.dtype),
                _meta_tensor(meta, "integrated_delta_up_m", device, pred.dtype),
            ],
            dim=1,
        )
        velocity_loss = criterion(pred_physical[:, body_indices], aux)
    elif velocity_weight > 0.0 and len(global_indices) == 3:
        aux = torch.stack(
            [
                _meta_tensor(meta, "integrated_delta_x_local_m", device, pred.dtype),
                _meta_tensor(meta, "integrated_delta_y_local_m", device, pred.dtype),
                _meta_tensor(meta, "integrated_delta_z_local_m", device, pred.dtype),
            ],
            dim=1,
        )
        velocity_loss = criterion(pred_physical[:, global_indices], aux)
    heading_indices = _indices(target_columns, ["delta_heading_deg"])
    if heading_weight > 0.0 and heading_indices:
        source = str(cfg.get("heading_integration_source", "gt_heading")).lower()
        key = "gyro_integrated_delta_heading_deg" if source == "gyro_z" else "integrated_delta_heading_deg"
        aux_heading = _meta_tensor(meta, key, device, pred.dtype)
        heading_loss = criterion(pred_physical[:, heading_indices[0]], aux_heading)
    return velocity_weight * velocity_loss + heading_weight * heading_loss, velocity_loss, heading_loss


def _physical_metrics(
    pred_batches: list[np.ndarray],
    target_batches: list[np.ndarray],
    target_stats: dict[str, list[float]] | None,
    target_layout: str,
    target_columns: list[str],
) -> dict[str, float]:
    if not pred_batches or not target_batches:
        return {"rmse": float("inf")}
    pred = _inverse_targets(np.concatenate(pred_batches, axis=0), target_stats, target_layout)
    target = _inverse_targets(np.concatenate(target_batches, axis=0), target_stats, target_layout)
    err = pred - target
    groups = _metric_groups(target_columns)
    return {
        "rmse": _component_rmse(err),
        "vector_rmse": float(np.sqrt(np.mean(np.sum(err**2, axis=1)))) if err.size else float("nan"),
        "position_rmse_m": _vector_rmse(err, groups["position"]),
        "velocity_rmse_mps": _vector_rmse(err, groups["velocity"]),
        "attitude_rmse_deg": _vector_rmse(err, groups["attitude"]),
    }


def _run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    target_layout: str,
    prediction_head: str,
    target_stats: dict[str, list[float]] | None,
    target_columns: list[str],
    train_cfg: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 0.0,
) -> tuple[dict[str, float], list[np.ndarray], list[np.ndarray]]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    signal_losses = []
    velocity_losses = []
    heading_losses = []
    preds = []
    targets = []
    for x, y, meta in loader:
        x = x.to(device)
        y = y.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        pred = _project_prediction(model(x), target_layout, prediction_head)
        signal_loss = criterion(pred, y)
        aux_loss, velocity_loss, heading_loss = (
            _integration_loss(pred, meta, target_stats, target_columns, criterion, train_cfg, device)
            if target_layout == "window"
            else (pred.new_tensor(0.0), pred.new_tensor(0.0), pred.new_tensor(0.0))
        )
        loss = signal_loss + aux_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training or validation loss during optimization")
        if is_train:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        signal_losses.append(float(signal_loss.detach().cpu()))
        velocity_losses.append(float(velocity_loss.detach().cpu()))
        heading_losses.append(float(heading_loss.detach().cpu()))
        if not is_train:
            preds.append(pred.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    metrics = {
        "loss": float(np.mean(losses)) if losses else float("inf"),
        "signal_loss": float(np.mean(signal_losses)) if signal_losses else float("inf"),
        "velocity_integration_loss": float(np.mean(velocity_losses)) if velocity_losses else 0.0,
        "heading_integration_loss": float(np.mean(heading_losses)) if heading_losses else 0.0,
    }
    return metrics, preds, targets


def _write_summary(
    run_dir: Path,
    config: dict[str, Any],
    best_epoch: int,
    best_metric: float,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float] | None,
) -> None:
    lines = ["# GT-Supervised Training Summary", ""]
    lines.append(f"- Experiment: `{config.get('experiment_name', 'gt_run')}`")
    lines.append(f"- Label mode: `{config.get('data', {}).get('label_mode', 'velocity_position')}`")
    lines.append(f"- Best epoch: `{best_epoch}`")
    lines.append(f"- Best metric: `{best_metric:.6g}`")
    lines.append("")
    lines.append("| split | loss | rmse | position rmse m | velocity rmse m/s | attitude rmse deg |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| val | {val_metrics.get('loss', float('nan')):.6g} | {val_metrics.get('rmse', float('nan')):.6g} | "
        f"{val_metrics.get('position_rmse_m', float('nan')):.6g} | {val_metrics.get('velocity_rmse_mps', float('nan')):.6g} | "
        f"{val_metrics.get('attitude_rmse_deg', float('nan')):.6g} |"
    )
    if test_metrics is not None:
        lines.append(
            f"| test | {test_metrics.get('test_loss', float('nan')):.6g} | {test_metrics.get('test_rmse', float('nan')):.6g} | "
            f"{test_metrics.get('test_position_rmse_m', float('nan')):.6g} | {test_metrics.get('test_velocity_rmse_mps', float('nan')):.6g} | "
            f"{test_metrics.get('test_attitude_rmse_deg', float('nan')):.6g} |"
        )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_safe_metrics(metrics: dict[str, float]) -> dict[str, float | None]:
    safe: dict[str, float | None] = {}
    for key, value in metrics.items():
        numeric = float(value)
        safe[key] = numeric if np.isfinite(numeric) else None
    return safe


def train_gt_supervised_model(config: dict[str, Any], config_path: str | Path | None = None) -> Path:
    """Train a GT-supervised inertial-odometry model and return the run directory."""
    seed_everything(int(config.get("seed", 0)))
    data_cfg = config["data"]
    train_cfg = config.get("training", {})
    label_mode = str(data_cfg.get("label_mode", "velocity_position"))
    evaluate_test = bool(train_cfg.get("evaluate_test", True)) and not bool(train_cfg.get("disable_test_evaluation", False))
    requested_splits = {
        "train": list(data_cfg.get("train_sequences", [])),
        "val": list(data_cfg.get("val_sequences", [])),
    }
    if evaluate_test:
        requested_splits["test"] = list(data_cfg.get("test_sequences", []))
    else:
        logger.info("SKIPS held-out test loading and automatic test evaluation (validation-only training)")
    actual_data_cfg = dict(data_cfg)
    if not evaluate_test:
        actual_data_cfg["load_splits"] = ["train", "validation"]
    loaders = make_smartpnt_gt_split_dataloaders(
        unified_navigation_frame(actual_data_cfg) if data_cfg.get("dataset_manifest") else data_cfg.get("aligned_root", data_cfg.get("aligned_paths")),
        split_sequences=requested_splits,
        skipped_sequences=_skipped_sequence_names(data_cfg.get("skipped_sequences")),
        window_size=int(data_cfg.get("window_size", 200)),
        stride=int(data_cfg.get("stride", 50)),
        batch_size=int(data_cfg.get("batch_size", 32)),
        label_mode=label_mode,
        normalize_features=bool(data_cfg.get("normalize_features", True)),
        normalize_targets=bool(data_cfg.get("normalize_targets", True)),
        num_workers=int(data_cfg.get("num_workers", 0)),
    )
    train_ds = loaders["train"].dataset
    val_ds = loaders["val"].dataset
    target_layout = train_ds.target_layout
    target_columns = train_ds.target_columns
    prediction_head = str(config.get("model", {}).get("prediction_head", train_cfg.get("prediction_head", "last")))

    model_cfg = dict(config.get("model", {}))
    model_cfg["in_channels"] = len(SMARTPNT_GT_IMU_COLUMNS)
    model_cfg["out_channels"] = len(target_columns)
    device = _device_from_config(str(train_cfg.get("device", "auto")))
    model = make_model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    criterion = _loss_fn(str(train_cfg.get("loss", "huber")))
    scheduler = warmup_cosine_scheduler(optimizer, train_cfg) if train_cfg.get("lr_scheduler") == "cosine_warmup" else None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.get("output", {}).get("runs_dir", "runs")) / f"{config.get('experiment_name', 'gt_run')}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "config.yaml")
    if config_path:
        (run_dir / "source_config_path.txt").write_text(str(config_path), encoding="utf-8")
    (run_dir / "provenance.json").write_text(json.dumps({"git_commit": git_commit(), **snapshot_unified_training_inputs(data_cfg, run_dir)}, indent=2), encoding="utf-8")
    metrics_logger = MetricsLogger(run_dir / "metrics.csv")

    checkpoint_metric = str(train_cfg.get("checkpoint_metric", "val_rmse"))
    best_metric = float("inf")
    best_epoch = 0
    stale_epochs = 0
    patience = int(train_cfg.get("early_stopping_patience", max(1, int(train_cfg.get("epochs", 1)))))
    best_val_metrics: dict[str, float] = {}
    logger.info(
        "GT training on {} with train={} val={} label_mode={} target_layout={}",
        device,
        len(train_ds),
        len(val_ds),
        label_mode,
        target_layout,
    )
    for epoch in range(1, int(train_cfg.get("epochs", 5)) + 1):
        train_epoch, _, _ = _run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            target_layout,
            prediction_head,
            train_ds.target_stats,
            target_columns,
            train_cfg,
            optimizer=optimizer,
            grad_clip=float(train_cfg.get("grad_clip_norm", 0.0)),
        )
        with torch.no_grad():
            val_epoch, val_preds, val_targets = _run_epoch(
                model,
                loaders["val"],
                criterion,
                device,
                target_layout,
                prediction_head,
                val_ds.target_stats,
                target_columns,
                train_cfg,
            )
        val_metrics = _physical_metrics(
            val_preds,
            val_targets,
            val_ds.target_stats,
            target_layout,
            target_columns,
        )
        row = {
            "epoch": float(epoch),
            "train_loss": train_epoch["loss"],
            "train_signal_loss": train_epoch["signal_loss"],
            "train_velocity_integration_loss": train_epoch["velocity_integration_loss"],
            "train_heading_integration_loss": train_epoch["heading_integration_loss"],
            "val_loss": val_epoch["loss"],
            "val_signal_loss": val_epoch["signal_loss"],
            "val_velocity_integration_loss": val_epoch["velocity_integration_loss"],
            "val_heading_integration_loss": val_epoch["heading_integration_loss"],
            "val_rmse": val_metrics["rmse"],
            "val_vector_rmse": val_metrics["vector_rmse"],
            "val_position_rmse_m": val_metrics["position_rmse_m"],
            "val_velocity_rmse_mps": val_metrics["velocity_rmse_mps"],
            "val_attitude_rmse_deg": val_metrics["attitude_rmse_deg"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        metrics_logger.log(row)
        monitor = float(row.get(checkpoint_metric, row["val_rmse"]))
        if not np.isfinite(monitor):
            raise FloatingPointError("Nonfinite validation selection metric during optimization")
        if scheduler is not None:
            scheduler.step()
        logger.info(
            "GT epoch {:03d}: train={:.5f} val={:.5f} rmse={:.5f}",
            epoch,
            train_epoch["loss"],
            val_epoch["loss"],
            row["val_rmse"],
        )
        if monitor < best_metric:
            best_metric = monitor
            best_epoch = epoch
            stale_epochs = 0
            best_val_metrics = {"loss": val_epoch["loss"], **val_metrics}
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_cfg,
                    "config": config,
                    "feature_stats": train_ds.feature_stats,
                    "target_stats": train_ds.target_stats,
                    "feature_columns": SMARTPNT_GT_IMU_COLUMNS,
                    "target_columns": target_columns,
                    "label_mode": label_mode,
                    "target_layout": target_layout,
                    "prediction_head": prediction_head,
                    "split_sequences": {
                        "train": list(data_cfg.get("train_sequences", [])),
                        "val": list(data_cfg.get("val_sequences", [])),
                        "test": list(data_cfg.get("test_sequences", [])),
                    },
                    "git_commit": git_commit(),
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "checkpoint_metric": checkpoint_metric,
                },
                run_dir / "best.ckpt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                logger.info("GT early stopping after {} stale epochs", stale_epochs)
                break

    test_metrics = None
    if "test" in loaders:
        checkpoint = torch.load(run_dir / "best.ckpt", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_ds = loaders["test"].dataset
        with torch.no_grad():
            test_epoch, test_preds, test_targets = _run_epoch(
                model,
                loaders["test"],
                criterion,
                device,
                target_layout,
                prediction_head,
                test_ds.target_stats,
                target_columns,
                train_cfg,
            )
        physical = _physical_metrics(
            test_preds,
            test_targets,
            test_ds.target_stats,
            target_layout,
            target_columns,
        )
        test_metrics = {
            "test_loss": test_epoch["loss"],
            "test_signal_loss": test_epoch["signal_loss"],
            "test_velocity_integration_loss": test_epoch["velocity_integration_loss"],
            "test_heading_integration_loss": test_epoch["heading_integration_loss"],
            "test_rmse": physical["rmse"],
            "test_vector_rmse": physical["vector_rmse"],
            "test_position_rmse_m": physical["position_rmse_m"],
            "test_velocity_rmse_mps": physical["velocity_rmse_mps"],
            "test_attitude_rmse_deg": physical["attitude_rmse_deg"],
        }
        (run_dir / "test_metrics.json").write_text(
            json.dumps(_json_safe_metrics(test_metrics), indent=2),
            encoding="utf-8",
        )
    _write_summary(run_dir, config, best_epoch, best_metric, best_val_metrics, test_metrics)
    logger.info("Finished GT-supervised training. Best checkpoint: {}", run_dir / "best.ckpt")
    return run_dir
