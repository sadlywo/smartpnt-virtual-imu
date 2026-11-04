from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from virtual_imu.training.dataset import (
    LOW_IMU_COLUMNS,
    TARGET_COLUMNS,
    IMUWindowDataset,
    select_low_motion_source,
)


def _frame(sequence_id: str, length: int = 32, low_sensor: str = "low-a") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    low = rng.normal(0.0, 1.0, size=(length, 6)).astype(np.float32)
    high = low + rng.normal(0.0, 0.05, size=(length, 6)).astype(np.float32)
    frame = pd.DataFrame(low, columns=LOW_IMU_COLUMNS)
    for index, column in enumerate(TARGET_COLUMNS):
        frame[column] = high[:, index]
    frame["timestamp_s"] = np.arange(length, dtype=float) * 0.01
    frame["sequence_id"] = sequence_id
    frame["low_sensor"] = low_sensor
    frame["training_eligible"] = True
    return frame


def test_low_motion_selector_finds_planted_interval_without_high_target_dependence():
    frame = _frame("af", 40)
    frame.loc[12:23, LOW_IMU_COLUMNS[:3]] = np.float32(0.001)
    frame.loc[12:23, LOW_IMU_COLUMNS[3:]] = np.asarray([0.0, 0.0, 9.81], dtype=np.float32)

    selected = select_low_motion_source(frame, window_size=12)
    changed = frame.copy()
    changed[TARGET_COLUMNS] = 1.0e9
    selected_after_target_change = select_low_motion_source(changed, window_size=12)

    assert selected.index.tolist() == list(range(12, 24))
    assert selected_after_target_change.index.tolist() == selected.index.tolist()
    assert selected.attrs["static_context_selection"]["sequence_id"] == "af"
    assert selected.attrs["static_context_selection"]["start_row"] == 12


def test_low_motion_selector_never_spans_sequence_boundaries():
    first = _frame("af", 8)
    second = _frame("as", 8)
    first.loc[:, LOW_IMU_COLUMNS[:3]] = 0.0
    second.loc[:, LOW_IMU_COLUMNS[:3]] = 0.0
    combined = pd.concat([first, second], ignore_index=True)

    with pytest.raises(ValueError, match="No sequence contains"):
        select_low_motion_source(combined, window_size=12)


def test_validation_dataset_reuses_exact_train_static_window():
    train = _frame("af", 24)
    validation = _frame("bf", 24)
    config = {
        "enabled": True,
        "columns": LOW_IMU_COLUMNS,
        "window_size": 8,
        "selection": "auto_low_motion",
        "reference_scope": "train_sensor",
    }
    train_dataset = IMUWindowDataset(
        train,
        window_size=8,
        stride=8,
        feature_columns=LOW_IMU_COLUMNS,
        static_context=config,
    )
    validation_dataset = IMUWindowDataset(
        validation,
        window_size=8,
        stride=8,
        feature_columns=LOW_IMU_COLUMNS,
        feature_stats=train_dataset.feature_stats,
        target_stats=train_dataset.target_stats,
        static_context=config,
        static_windows=train_dataset.static_windows,
        static_window_metadata=train_dataset.static_window_metadata,
        allow_static_context_build=False,
    )

    assert np.array_equal(
        validation_dataset.static_windows["low-a"],
        train_dataset.static_windows["low-a"],
    )
    assert validation_dataset.static_window_metadata["low-a"]["sequence_id"] == "af"


def test_validation_dataset_fails_when_train_sensor_context_is_missing():
    validation = _frame("bf", 24, low_sensor="unseen")
    config = {
        "enabled": True,
        "columns": LOW_IMU_COLUMNS,
        "window_size": 8,
        "selection": "auto_low_motion",
        "reference_scope": "train_sensor",
    }

    with pytest.raises(ValueError, match="Missing precomputed static context"):
        IMUWindowDataset(
            validation,
            window_size=8,
            stride=8,
            feature_columns=LOW_IMU_COLUMNS,
            static_context=config,
            static_windows={},
            allow_static_context_build=False,
        )
