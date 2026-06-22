from __future__ import annotations

from pathlib import Path

import pandas as pd

from virtual_imu.inference.virtual_imu_generator import VirtualIMUGenerator
from virtual_imu.sim.paired_dataset import generate_paired_dataset
from virtual_imu.training.trainer import train_model


def test_training_and_inference_smoke(tmp_path):
    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs"
    cfg = {
        "seed": 2,
        "output_dir": str(data_dir),
        "rate_hz": 30.0,
        "splits": {
            "train": {"duration_s": 5.0, "sequence_id": "tr"},
            "val": {"duration_s": 4.0, "sequence_id": "va"},
            "test": {"duration_s": 3.0, "sequence_id": "te"},
        },
        "imu_error": {"gyro_noise_std": 0.002, "accel_noise_std": 0.02},
    }
    generate_paired_dataset(cfg)
    train_cfg = {
        "seed": 2,
        "experiment_name": "pytest",
        "data": {
            "train_path": str(data_dir / "train.parquet"),
            "val_path": str(data_dir / "val.parquet"),
            "window_size": 32,
            "stride": 16,
            "batch_size": 4,
            "num_workers": 0,
        },
        "model": {
            "type": "tcn",
            "target_mode": "competitor_residual",
            "hidden_channels": 8,
            "levels": 1,
            "kernel_size": 3,
            "dropout": 0.0,
            "zero_init_head": True,
        },
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "loss": "huber",
            "checkpoint_metric": "val_score",
            "competitor_baseline": {"enabled": True, "mode": "blend", "fir_taps": 5},
            "baseline_margin_weight": 0.1,
            "validation_score": {
                "relative_to_competitor": True,
                "relative_gyro_rmse_weight": 1.0,
                "relative_accel_rmse_weight": 0.5,
                "relative_preintegration_weight": 0.5,
            },
            "device": "cpu",
            "early_stopping_patience": 2,
        },
        "output": {"runs_dir": str(runs_dir)},
    }
    run_dir = train_model(train_cfg)
    ckpt = Path(run_dir) / "best.ckpt"
    assert ckpt.exists()
    metrics = pd.read_csv(Path(run_dir) / "metrics.csv")
    assert "val_relative_gyro_rmse" in metrics.columns
    assert "val_competitor_gyro_rmse" in metrics.columns
    out = tmp_path / "pred.parquet"
    VirtualIMUGenerator(ckpt, device="cpu").run(data_dir / "test.parquet", out)
    assert out.exists()
