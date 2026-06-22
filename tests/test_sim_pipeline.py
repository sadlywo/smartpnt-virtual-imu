from __future__ import annotations

import pandas as pd

from virtual_imu.sim.paired_dataset import generate_paired_dataset


def test_sim_pipeline_generates_standard_columns(tmp_path):
    cfg = {
        "seed": 1,
        "output_dir": str(tmp_path),
        "rate_hz": 20.0,
        "splits": {"train": {"duration_s": 3.0, "sequence_id": "test"}},
        "imu_error": {"gyro_noise_std": 0.001, "accel_noise_std": 0.01},
    }
    paths = generate_paired_dataset(cfg)
    df = pd.read_parquet(paths["train"])
    expected = {
        "timestamp_s",
        "gx_low",
        "gy_low",
        "gz_low",
        "ax_low",
        "ay_low",
        "az_low",
        "gx_high",
        "gy_high",
        "gz_high",
        "ax_high",
        "ay_high",
        "az_high",
        "temperature_c",
        "motion_label",
        "sequence_id",
    }
    assert expected.issubset(df.columns)
    assert len(df) == 60
