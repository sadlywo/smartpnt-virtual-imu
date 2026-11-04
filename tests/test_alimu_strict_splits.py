from __future__ import annotations

import pandas as pd

from virtual_imu.io.alimu_splits import make_strict_split


def _row(sequence_id: str, sensor: str, motion: str) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp_s": 0.0,
        "sequence_id": sequence_id,
        "low_sensor": sensor,
        "motion_label": motion,
        "source_file": f"{sequence_id}.csv",
    }
    for col in ["gx", "gy", "gz", "ax", "ay", "az"]:
        row[f"{col}_low"] = 1.0
        row[f"{col}_high"] = 2.0
    return row


def test_make_strict_split_leaves_sensor_out_by_sequence():
    df = pd.DataFrame(
        [
            _row("bmx_dyn", "BMX160", "dynamic"),
            _row("bmx_stat", "BMX160", "stationary"),
            _row("mpu_dyn", "MPU9250", "dynamic"),
            _row("mpu_stat", "MPU9250", "stationary"),
            _row("lsm_dyn", "LSM9DS1", "dynamic"),
            _row("lsm_stat", "LSM9DS1", "stationary"),
        ]
    )

    split = make_strict_split(df, mode="leave_sensor_out", holdout="MPU9250", val_fraction=0.25, seed=7)

    assert set(split["test"]["low_sensor"]) == {"MPU9250"}
    assert "MPU9250" not in set(split["train"]["low_sensor"])
    assert "MPU9250" not in set(split["val"]["low_sensor"])
    assert set(split["train"]["sequence_id"]).isdisjoint(split["val"]["sequence_id"])
    assert set(split["train"]["sequence_id"]).isdisjoint(split["test"]["sequence_id"])


def test_make_strict_split_leaves_motion_out_by_sequence():
    df = pd.DataFrame(
        [
            _row("bmx_dyn", "BMX160", "dynamic"),
            _row("bmx_stat", "BMX160", "stationary"),
            _row("mpu_dyn", "MPU9250", "dynamic"),
            _row("mpu_stat", "MPU9250", "stationary"),
        ]
    )

    split = make_strict_split(df, mode="leave_motion_out", holdout="stationary", val_fraction=0.5, seed=3)

    assert set(split["test"]["motion_label"]) == {"stationary"}
    assert "stationary" not in set(split["train"]["motion_label"])
    assert "stationary" not in set(split["val"]["motion_label"])
    assert set(split["train"]["sequence_id"]).isdisjoint(split["val"]["sequence_id"])
    assert set(split["train"]["sequence_id"]).isdisjoint(split["test"]["sequence_id"])
