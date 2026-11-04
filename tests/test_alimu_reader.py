from __future__ import annotations

import pandas as pd

from virtual_imu.io.alimu_reader import prepare_alimu_dataset, read_alimu_csv


def test_read_alimu_real_csv_maps_low_and_high_columns(tmp_path):
    source = tmp_path / "Real" / "IMU1_BMX160" / "real_sensor_data_dyn_01_BMX160_VN100.csv"
    source.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "BMX160_ax": [1.0, 2.0],
            "BMX160_ay": [3.0, 4.0],
            "BMX160_az": [5.0, 6.0],
            "BMX160_gx": [10.0, 20.0],
            "BMX160_gy": [30.0, 40.0],
            "BMX160_gz": [50.0, 60.0],
            "BMX160_mx": [0.1, 0.2],
            "BMX160_my": [0.3, 0.4],
            "BMX160_mz": [0.5, 0.6],
            "VN100_ax": [7.0, 8.0],
            "VN100_ay": [9.0, 10.0],
            "VN100_az": [11.0, 12.0],
            "VN100_gx": [70.0, 80.0],
            "VN100_gy": [90.0, 100.0],
            "VN100_gz": [110.0, 120.0],
            "VN100_mx": [0.7, 0.8],
            "VN100_my": [0.9, 1.0],
            "VN100_mz": [1.1, 1.2],
        }
    ).to_csv(source, index=False)

    parsed, metadata = read_alimu_csv(source, raw_root=tmp_path, sample_rate_hz=40.0, gyro_unit="deg_per_s")

    assert list(parsed.columns) == [
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
        "source_kind",
        "low_sensor",
        "high_sensor",
        "source_file",
    ]
    assert parsed["timestamp_s"].tolist() == [0.0, 0.025]
    assert parsed["sequence_id"].nunique() == 1
    assert parsed["low_sensor"].unique().tolist() == ["BMX160"]
    assert parsed["high_sensor"].unique().tolist() == ["VN100"]
    assert parsed.loc[0, "motion_label"] == "dynamic"
    assert parsed.loc[0, "gx_low"] == pd.Series([10.0]).mul(3.141592653589793 / 180.0).iloc[0]
    assert parsed.loc[1, "az_high"] == 12.0
    assert metadata["rows"] == 2
    assert metadata["source_kind"] == "real"


def test_prepare_alimu_dataset_writes_sequence_level_splits(tmp_path):
    raw_root = tmp_path / "raw"
    sensor_dir = raw_root / "Real" / "IMU1_BMX160"
    sensor_dir.mkdir(parents=True)
    for idx in range(5):
        pd.DataFrame(
            {
                "BMX160_ax": [float(idx), float(idx + 1)],
                "BMX160_ay": [1.0, 1.1],
                "BMX160_az": [9.0, 9.1],
                "BMX160_gx": [0.1, 0.2],
                "BMX160_gy": [0.3, 0.4],
                "BMX160_gz": [0.5, 0.6],
                "VN100_ax": [float(idx + 2), float(idx + 3)],
                "VN100_ay": [2.0, 2.1],
                "VN100_az": [8.0, 8.1],
                "VN100_gx": [0.7, 0.8],
                "VN100_gy": [0.9, 1.0],
                "VN100_gz": [1.1, 1.2],
            }
        ).to_csv(sensor_dir / f"real_sensor_data_dyn_{idx:02d}_BMX160_VN100.csv", index=False)

    processed_root = tmp_path / "processed"
    result = prepare_alimu_dataset(
        {
            "raw_root": str(raw_root),
            "processed_root": str(processed_root),
            "include": {"real": True, "synthetic": False},
            "real_rate_hz": 40.0,
            "split": {"train": 0.6, "val": 0.2, "test": 0.2, "seed": 123},
        }
    )

    assert result["train_path"] == processed_root / "train.parquet"
    assert result["val_path"] == processed_root / "val.parquet"
    assert result["test_path"] == processed_root / "test.parquet"
    train = pd.read_parquet(result["train_path"])
    val = pd.read_parquet(result["val_path"])
    test = pd.read_parquet(result["test_path"])
    assert len(set(train["sequence_id"]) & set(val["sequence_id"])) == 0
    assert len(set(train["sequence_id"]) & set(test["sequence_id"])) == 0
    assert len(set(val["sequence_id"]) & set(test["sequence_id"])) == 0
    assert len(result["manifest"]["files"]) == 5
    assert result["manifest"]["splits"]["train"]["sequences"] == 3
    assert result["manifest"]["splits"]["val"]["sequences"] == 1
    assert result["manifest"]["splits"]["test"]["sequences"] == 1
