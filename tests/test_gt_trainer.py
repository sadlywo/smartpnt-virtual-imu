from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from virtual_imu.training.gt_trainer import train_gt_supervised_model


def _gt_frame(sequence_id: str, offset: float) -> pd.DataFrame:
    n = 64
    t = np.arange(n, dtype=float) * 0.01
    heading = (offset + np.linspace(0.0, 18.0, n)) % 360.0
    x = offset + 2.0 * t
    y = -offset + 1.5 * t
    z = 0.1 * np.sin(t)
    return pd.DataFrame(
        {
            "timestamp_s": t + offset * 10.0,
            "sequence_id": [sequence_id] * n,
            "gx_radps": np.gradient(heading, t) * np.pi / 180.0,
            "gy_radps": np.sin(t),
            "gz_radps": np.cos(t) * 0.1,
            "ax_mps2": -9.81 + 0.01 * np.sin(t),
            "ay_mps2": 0.1 * np.cos(t),
            "az_mps2": 0.02 * np.sin(2.0 * t),
            "gt_v_east_mps": np.gradient(x, t),
            "gt_v_north_mps": np.gradient(y, t),
            "gt_v_up_mps": np.gradient(z, t),
            "gt_x_local_m": x,
            "gt_y_local_m": y,
            "gt_z_local_m": z,
            "gt_heading_deg": heading,
            "gt_pitch_deg": 0.1 * np.sin(t),
            "gt_roll_deg": -0.2 * np.cos(t),
        }
    )


def test_gt_supervised_trainer_writes_checkpoint_metrics_and_test_report(tmp_path):
    data_root = tmp_path / "aligned"
    data_root.mkdir()
    for sequence_id, offset in [("train_seq", 1.0), ("val_seq", 2.0), ("test_seq", 3.0)]:
        _gt_frame(sequence_id, offset).to_parquet(data_root / f"{sequence_id}_imu_gt_aligned.parquet", index=False)

    cfg = {
        "seed": 3,
        "experiment_name": "pytest_gt",
        "data": {
            "aligned_root": str(data_root),
            "label_mode": "body_pose_delta",
            "window_size": 16,
            "stride": 8,
            "batch_size": 4,
            "num_workers": 0,
            "train_sequences": ["train_seq"],
            "val_sequences": ["val_seq"],
            "test_sequences": ["test_seq"],
        },
        "model": {
            "type": "tcn",
            "hidden_channels": 8,
            "levels": 1,
            "kernel_size": 3,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "loss": "huber",
            "device": "cpu",
            "checkpoint_metric": "val_rmse",
            "velocity_integration_weight": 0.1,
            "heading_integration_weight": 0.1,
        },
        "output": {"runs_dir": str(tmp_path / "runs")},
    }

    run_dir = train_gt_supervised_model(cfg)

    assert (run_dir / "best.ckpt").exists()
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "test_metrics.json").exists()
    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert {"train_loss", "val_loss", "val_rmse"}.issubset(metrics.columns)
    ckpt = torch.load(run_dir / "best.ckpt", map_location="cpu")
    assert ckpt["label_mode"] == "body_pose_delta"
    assert ckpt["target_columns"] == [
        "delta_forward_m",
        "delta_right_m",
        "delta_up_m",
        "delta_heading_deg",
        "delta_pitch_deg",
        "delta_roll_deg",
    ]
    assert "train_velocity_integration_loss" in metrics.columns
    assert "val_heading_integration_loss" in metrics.columns
    test_metrics = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))
    assert "test_rmse" in test_metrics
