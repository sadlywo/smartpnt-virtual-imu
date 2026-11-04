from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from virtual_imu.inference.virtual_imu_generator import VIRTUAL_COLUMNS, VirtualIMUGenerator
from virtual_imu.models import make_model
from virtual_imu.training.dataset import LOW_IMU_COLUMNS, TARGET_COLUMNS


def _checkpoint(path: Path, static_windows: dict[str, np.ndarray]) -> Path:
    model_config = {
        "type": "bias_drift_tcn",
        "in_channels": 6,
        "out_channels": 6,
        "static_channels": 6,
        "hidden_channels": 4,
        "levels": 1,
        "kernel_size": 3,
        "dropout": 0.0,
        "bias_branch_mode": "offset",
    }
    model = make_model(model_config)
    torch.save(
        {
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "feature_columns": LOW_IMU_COLUMNS,
            "feature_stats": {"columns": LOW_IMU_COLUMNS, "mean": [0.0] * 6, "std": [1.0] * 6},
            "target_stats": {"columns": TARGET_COLUMNS, "mean": [0.0] * 6, "std": [1.0] * 6},
            "target_mode": "residual",
            "static_context": {
                "enabled": True,
                "columns": LOW_IMU_COLUMNS,
                "window_size": 8,
                "reference_scope": "train_sensor",
            },
            "static_windows": static_windows,
            "config": {"data": {"window_size": 4, "stride": 2}},
        },
        path,
    )
    return path


def _frame(sensor: str = "low-a") -> pd.DataFrame:
    frame = pd.DataFrame(np.zeros((8, 6), dtype=np.float32), columns=LOW_IMU_COLUMNS)
    frame["timestamp_s"] = np.arange(8) * 0.01
    frame["sequence_id"] = "bs"
    frame["low_sensor"] = sensor
    return frame


def test_train_sensor_inference_uses_checkpoint_context(monkeypatch, tmp_path):
    checkpoint = _checkpoint(tmp_path / "model.ckpt", {"low-a": np.ones((8, 6), dtype=np.float32)})

    def fail_if_rebuilt(*args, **kwargs):
        raise AssertionError("inference attempted to rebuild static context")

    monkeypatch.setattr(
        "virtual_imu.inference.virtual_imu_generator.build_static_context_windows",
        fail_if_rebuilt,
    )
    result = VirtualIMUGenerator(checkpoint, device="cpu").run_frame(_frame())

    assert result[VIRTUAL_COLUMNS].shape == (8, 6)
    assert result["low_sensor"].tolist() == _frame()["low_sensor"].tolist()
    assert np.isfinite(result[VIRTUAL_COLUMNS].to_numpy()).all()


def test_train_sensor_inference_fails_for_missing_sensor_context(tmp_path):
    checkpoint = _checkpoint(tmp_path / "model.ckpt", {"low-a": np.ones((8, 6), dtype=np.float32)})

    with pytest.raises(ValueError, match="missing train-derived static context"):
        VirtualIMUGenerator(checkpoint, device="cpu").run_frame(_frame("unseen"))
