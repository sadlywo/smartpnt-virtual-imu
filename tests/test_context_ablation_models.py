from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from virtual_imu.inference.virtual_imu_generator import VirtualIMUGenerator
from virtual_imu.models import make_model
from virtual_imu.sim.paired_dataset import generate_paired_dataset
from virtual_imu.training.dataset import IMUWindowDataset
from virtual_imu.training.trainer import train_model


def test_window_dataset_uses_configured_context_feature_columns():
    frame = pd.DataFrame(
        {
            "timestamp_s": [0.0, 0.01, 0.02, 0.03],
            "gx_low": [0.0, 0.1, 0.2, 0.3],
            "gy_low": [0.0, 0.0, 0.0, 0.0],
            "gz_low": [0.0, 0.0, 0.0, 0.0],
            "ax_low": [1.0, 1.0, 1.0, 1.0],
            "ay_low": [0.0, 0.0, 0.0, 0.0],
            "az_low": [-9.8, -9.8, -9.8, -9.8],
            "temperature_c": [25.0, 25.0, 25.0, 25.0],
            "spec_gyro_noise_density": [0.01, 0.01, 0.01, 0.01],
            "static_bias_gx": [0.02, 0.02, 0.02, 0.02],
            "gx_high": [0.0, 0.0, 0.0, 0.0],
            "gy_high": [0.0, 0.0, 0.0, 0.0],
            "gz_high": [0.0, 0.0, 0.0, 0.0],
            "ax_high": [0.0, 0.0, 0.0, 0.0],
            "ay_high": [0.0, 0.0, 0.0, 0.0],
            "az_high": [-9.8, -9.8, -9.8, -9.8],
            "sequence_id": ["s"] * 4,
        }
    )
    feature_columns = [
        "gx_low",
        "gy_low",
        "gz_low",
        "ax_low",
        "ay_low",
        "az_low",
        "temperature_c",
        "spec_gyro_noise_density",
        "static_bias_gx",
    ]

    dataset = IMUWindowDataset(frame, window_size=4, stride=4, feature_columns=feature_columns)

    x, y = dataset[0]
    assert tuple(x.shape) == (len(feature_columns), 4)
    assert tuple(y.shape) == (6, 4)
    assert dataset.feature_columns == feature_columns


def test_film_tcn_factory_outputs_channel_first_windows():
    model = make_model(
        {
            "type": "film_tcn",
            "in_channels": 12,
            "out_channels": 6,
            "context_channels": 5,
            "hidden_channels": 16,
            "levels": 2,
        }
    )

    y = model(torch.randn(2, 12, 32))

    assert tuple(y.shape) == (2, 6, 32)


def test_training_and_inference_preserve_context_feature_columns(tmp_path):
    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs"
    generate_paired_dataset(
        {
            "seed": 3,
            "output_dir": str(data_dir),
            "rate_hz": 30.0,
            "splits": {
                "train": {"duration_s": 5.0, "sequence_id": "tr"},
                "val": {"duration_s": 4.0, "sequence_id": "va"},
                "test": {"duration_s": 3.0, "sequence_id": "te"},
            },
            "imu_error": {"gyro_noise_std": 0.002, "accel_noise_std": 0.02},
        }
    )
    feature_columns = [
        "gx_low",
        "gy_low",
        "gz_low",
        "ax_low",
        "ay_low",
        "az_low",
        "temperature_c",
        "spec_gyro_noise_density",
    ]
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.parquet"
        frame = pd.read_parquet(path)
        frame["spec_gyro_noise_density"] = 0.01
        frame.to_parquet(path, index=False)
    train_cfg = {
        "seed": 3,
        "experiment_name": "pytest_context",
        "data": {
            "train_path": str(data_dir / "train.parquet"),
            "val_path": str(data_dir / "val.parquet"),
            "feature_columns": feature_columns,
            "window_size": 32,
            "stride": 16,
            "batch_size": 4,
            "num_workers": 0,
        },
        "model": {
            "type": "tcn",
            "target_mode": "residual",
            "hidden_channels": 8,
            "levels": 1,
            "kernel_size": 3,
            "dropout": 0.0,
        },
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "loss": "huber",
            "checkpoint_metric": "val_loss",
            "device": "cpu",
            "early_stopping_patience": 2,
        },
        "output": {"runs_dir": str(runs_dir)},
    }

    run_dir = train_model(train_cfg)
    checkpoint = Path(run_dir) / "best.ckpt"
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["feature_columns"] == feature_columns
    out = tmp_path / "pred.parquet"
    VirtualIMUGenerator(checkpoint, device="cpu").run(data_dir / "test.parquet", out)

    assert out.exists()
