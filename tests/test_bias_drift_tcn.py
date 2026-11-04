from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from virtual_imu.inference.virtual_imu_generator import VirtualIMUGenerator
from virtual_imu.models import make_model
from virtual_imu.sim.paired_dataset import generate_paired_dataset
from virtual_imu.training.dataset import IMUWindowDataset
from virtual_imu.training.trainer import train_model


def _paired_frame() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        rows.append(
            {
                "timestamp_s": idx * 0.01,
                "gx_low": float(idx),
                "gy_low": 0.1,
                "gz_low": 0.2,
                "ax_low": 0.3,
                "ay_low": 0.4,
                "az_low": -9.8,
                "temperature_c": 25.0,
                "gx_high": float(idx) * 0.5,
                "gy_high": 0.0,
                "gz_high": 0.0,
                "ax_high": 0.0,
                "ay_high": 0.0,
                "az_high": -9.7,
                "sequence_id": "static" if idx < 8 else "dynamic",
                "motion_label": "stationary" if idx < 8 else "dynamic",
                "low_sensor": "S1",
            }
        )
    return pd.DataFrame(rows)


def test_dataset_returns_long_static_context_window():
    dataset = IMUWindowDataset(
        _paired_frame(),
        window_size=4,
        stride=4,
        target_mode="residual",
        static_context={"enabled": True, "window_size": 6},
    )

    x, y, static_context = dataset[0]

    assert tuple(x.shape) == (7, 4)
    assert tuple(y.shape) == (6, 4)
    assert tuple(static_context.shape) == (7, 6)


def test_bias_drift_tcn_factory_supports_branch_ablation():
    x = torch.randn(2, 7, 32)
    static_context = torch.randn(2, 7, 128)
    for use_bias, use_drift in [(True, False), (False, True), (True, True)]:
        model = make_model(
            {
                "type": "bias_drift_tcn",
                "in_channels": 7,
                "out_channels": 6,
                "static_channels": 7,
                "hidden_channels": 16,
                "levels": 2,
                "use_bias_branch": use_bias,
                "use_drift_branch": use_drift,
            }
        )
        y = model(x, static_context if use_drift else None)
        assert tuple(y.shape) == (2, 6, 32)


def test_constrained_bias_branch_outputs_window_constant_six_axis_offset():
    model = make_model(
        {
            "type": "bias_drift_tcn",
            "in_channels": 7,
            "out_channels": 6,
            "hidden_channels": 16,
            "levels": 2,
            "use_bias_branch": True,
            "use_drift_branch": False,
            "bias_branch_mode": "offset",
        }
    )

    y = model(torch.randn(2, 7, 32))

    assert tuple(y.shape) == (2, 6, 32)
    bias_offset = model.last_aux["bias_offset"]
    assert tuple(bias_offset.shape) == (2, 6, 1)


def test_dataset_returns_bias_auxiliary_target_for_static_windows():
    dataset = IMUWindowDataset(
        _paired_frame(),
        window_size=4,
        stride=4,
        target_mode="residual",
        bias_auxiliary={"enabled": True},
    )

    x, y, bias_target, bias_mask = dataset[0]

    assert tuple(x.shape) == (7, 4)
    assert tuple(y.shape) == (6, 4)
    assert tuple(bias_target.shape) == (6, 1)
    assert float(bias_mask.item()) == 1.0


def test_training_and_inference_with_static_context(tmp_path):
    data_dir = tmp_path / "data"
    runs_dir = tmp_path / "runs"
    generate_paired_dataset(
        {
            "seed": 13,
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
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}.parquet"
        frame = pd.read_parquet(path)
        frame["low_sensor"] = "SIM"
        frame["motion_label"] = "stationary"
        frame.to_parquet(path, index=False)
    cfg = {
        "seed": 13,
        "experiment_name": "pytest_bias_drift",
        "data": {
            "train_path": str(data_dir / "train.parquet"),
            "val_path": str(data_dir / "val.parquet"),
            "window_size": 32,
            "stride": 16,
            "batch_size": 4,
            "num_workers": 0,
            "static_context": {"enabled": True, "window_size": 48},
        },
        "model": {
            "type": "bias_drift_tcn",
            "target_mode": "residual",
            "hidden_channels": 8,
            "levels": 1,
            "kernel_size": 3,
            "dropout": 0.0,
            "static_channels": 7,
            "use_bias_branch": True,
            "use_drift_branch": True,
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

    run_dir = train_model(cfg)
    checkpoint = Path(run_dir) / "best.ckpt"
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["static_context"]["enabled"] is True
    out = tmp_path / "pred.parquet"
    VirtualIMUGenerator(checkpoint, device="cpu").run(data_dir / "test.parquet", out)

    assert out.exists()
