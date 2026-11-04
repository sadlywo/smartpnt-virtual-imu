from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from virtual_imu.inference.virtual_imu_generator import VIRTUAL_COLUMNS, VirtualIMUGenerator
from virtual_imu.models import make_model
from virtual_imu.training import trainer
from virtual_imu.training.dataset import (
    LOW_IMU_COLUMNS,
    TARGET_COLUMNS,
    IMUWindowDataset,
    build_static_context_windows,
)

TRAIN_GLOBAL_STATIC_CONTEXT_KEY = "__train_global__"


def _frame(sequence_id: str, sensor: str, low_value: float, length: int = 8) -> pd.DataFrame:
    low = np.full((length, 6), low_value, dtype=np.float32)
    frame = pd.DataFrame(low, columns=LOW_IMU_COLUMNS)
    for column in TARGET_COLUMNS:
        frame[column] = low_value
    frame["timestamp_s"] = np.arange(length, dtype=float) * 0.01
    frame["temperature_c"] = 25.0
    frame["sequence_id"] = sequence_id
    frame["low_sensor"] = sensor
    return frame


def _static_context() -> dict:
    return {
        "enabled": True,
        "columns": LOW_IMU_COLUMNS,
        "window_size": 8,
        "selection": "auto_low_motion",
        "reference_scope": "train_global",
    }


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
            "static_context": _static_context(),
            "static_windows": static_windows,
            "config": {"data": {"window_size": 4, "stride": 2}},
        },
        path,
    )
    return path


def test_train_global_builds_one_low_motion_window_with_source_metadata():
    train = pd.concat(
        [_frame("train-a", "BMX160", 1.0), _frame("train-b", "LSM9DS1", 0.0)],
        ignore_index=True,
    )
    metadata: dict[str, dict] = {}

    windows = build_static_context_windows(
        train,
        _static_context(),
        {"columns": LOW_IMU_COLUMNS, "mean": [0.0] * 6, "std": [1.0] * 6},
        metadata_out=metadata,
    )

    assert list(windows) == [TRAIN_GLOBAL_STATIC_CONTEXT_KEY]
    assert windows[TRAIN_GLOBAL_STATIC_CONTEXT_KEY].shape == (8, 6)
    assert np.array_equal(windows[TRAIN_GLOBAL_STATIC_CONTEXT_KEY], np.zeros((8, 6), dtype=np.float32))
    assert metadata[TRAIN_GLOBAL_STATIC_CONTEXT_KEY]["source_sensor"] == "LSM9DS1"
    assert metadata[TRAIN_GLOBAL_STATIC_CONTEXT_KEY]["sequence_id"] == "train-b"
    assert metadata[TRAIN_GLOBAL_STATIC_CONTEXT_KEY]["reference_scope"] == "train_global"


def test_train_global_low_motion_window_is_sensor_pure_for_shared_sequence_id():
    train = pd.concat(
        [_frame("shared", "BMX160", 0.0), _frame("shared", "LSM9DS1", 1.0)],
        ignore_index=True,
    )
    metadata: dict[str, dict] = {}

    windows = build_static_context_windows(
        train,
        _static_context(),
        {"columns": LOW_IMU_COLUMNS, "mean": [0.0] * 6, "std": [1.0] * 6},
        metadata_out=metadata,
    )

    assert np.array_equal(windows[TRAIN_GLOBAL_STATIC_CONTEXT_KEY], np.zeros((8, 6), dtype=np.float32))
    assert metadata[TRAIN_GLOBAL_STATIC_CONTEXT_KEY]["source_sensor"] == "BMX160"
    assert metadata[TRAIN_GLOBAL_STATIC_CONTEXT_KEY]["sequence_id"] == "shared"


def test_train_global_dataset_reuses_saved_global_window_without_rebuilding(monkeypatch):
    saved_window = np.ones((8, 6), dtype=np.float32)

    monkeypatch.setattr(
        "virtual_imu.training.dataset.build_static_context_windows",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dataset rebuilt static context")),
    )
    dataset = IMUWindowDataset(
        _frame("unseen", "MPU9250", 2.0),
        window_size=4,
        stride=4,
        feature_columns=LOW_IMU_COLUMNS,
        static_context=_static_context(),
        static_windows={TRAIN_GLOBAL_STATIC_CONTEXT_KEY: saved_window},
        allow_static_context_build=False,
    )

    _, _, static = dataset[0]
    assert np.array_equal(static.numpy(), saved_window.T)


def test_train_global_dataset_requires_explicit_opt_in_to_build_static_context():
    with pytest.raises(ValueError, match="allow_train_global_static_context_build"):
        IMUWindowDataset(
            _frame("unseen", "MPU9250", 2.0),
            window_size=4,
            stride=4,
            feature_columns=LOW_IMU_COLUMNS,
            static_context=_static_context(),
        )


def test_trainer_passes_train_global_context_to_validation_without_building(monkeypatch):
    train = _frame("train", "BMX160", 0.0)
    validation = _frame("validation", "MPU9250", 1.0)
    constructed: list[dict] = []
    datasets: list[RecordingDataset] = []

    class StopAfterValidationError(RuntimeError):
        pass

    class RecordingDataset:
        def __init__(self, *args, **kwargs) -> None:
            constructed.append(kwargs)
            datasets.append(self)
            self.feature_stats = {"columns": LOW_IMU_COLUMNS, "mean": [0.0] * 6, "std": [1.0] * 6}
            self.target_stats = {"columns": TARGET_COLUMNS, "mean": [0.0] * 6, "std": [1.0] * 6}
            self.static_windows = {TRAIN_GLOBAL_STATIC_CONTEXT_KEY: np.zeros((8, 6), dtype=np.float32)}
            self.static_window_metadata = {TRAIN_GLOBAL_STATIC_CONTEXT_KEY: {"source_sensor": "BMX160"}}
            if len(constructed) == 2:
                raise StopAfterValidationError()

    monkeypatch.setattr(trainer, "_load_training_frames", lambda data_cfg: (train, validation, None))
    monkeypatch.setattr(trainer, "_add_system_id_competitor", lambda train, val, cfg: (train, val, None, {}))
    monkeypatch.setattr(trainer, "IMUWindowDataset", RecordingDataset)

    with pytest.raises(StopAfterValidationError):
        trainer.train_model(
            {
                "data": {
                    "window_size": 4,
                    "stride": 4,
                    "static_context": _static_context(),
                },
                "model": {"type": "bias_drift_tcn", "target_mode": "residual"},
            }
        )

    validation_kwargs = constructed[1]
    assert constructed[0]["allow_train_global_static_context_build"] is True
    assert validation_kwargs.get("allow_train_global_static_context_build", False) is False
    assert validation_kwargs["static_windows"] is datasets[0].static_windows
    assert validation_kwargs["static_window_metadata"] is datasets[0].static_window_metadata
    assert validation_kwargs["allow_static_context_build"] is False


def test_train_global_inference_reuses_checkpoint_global_window_for_unseen_sensor(monkeypatch, tmp_path):
    checkpoint = _checkpoint(
        tmp_path / "model.ckpt", {TRAIN_GLOBAL_STATIC_CONTEXT_KEY: np.ones((8, 6), dtype=np.float32)}
    )
    monkeypatch.setattr(
        "virtual_imu.inference.virtual_imu_generator.build_static_context_windows",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inference rebuilt static context")),
    )

    result = VirtualIMUGenerator(checkpoint, device="cpu").run_frame(_frame("unseen", "MPU9250", 0.0))

    assert result[VIRTUAL_COLUMNS].shape == (8, 6)


def test_train_global_inference_fails_when_checkpoint_global_window_is_missing(tmp_path):
    checkpoint = _checkpoint(tmp_path / "model.ckpt", {})

    with pytest.raises(ValueError, match="missing train-derived static context"):
        VirtualIMUGenerator(checkpoint, device="cpu").run_frame(_frame("unseen", "MPU9250", 0.0))
