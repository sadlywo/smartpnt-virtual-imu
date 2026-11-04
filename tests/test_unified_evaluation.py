import numpy as np
import pandas as pd
import pytest

from virtual_imu.eval.reports import paired_device_metrics


def test_physical_errors_are_reported_per_reference_device():
    frames = []
    for dataset, reference, error in (("alimu", "VN100", 0.01), ("penncosyvio", "ADIS16488", 1.0)):
        frame = pd.DataFrame(
            {
                "dataset_id": [dataset] * 100,
                "low_sensor": "MEMS",
                "high_sensor": reference,
                "timestamp_s": np.arange(100) / 40,
            }
        )
        for axis in ("gx", "gy", "gz", "ax", "ay", "az"):
            frame[f"{axis}_high"] = np.sin(np.arange(100) / 10)
            frame[f"{axis}_low"] = frame[f"{axis}_high"] + error
            frame[f"{axis}_virtual"] = frame[f"{axis}_high"] + error / 2
        frames.append(frame)
    groups = paired_device_metrics(pd.concat(frames, ignore_index=True))
    assert len(groups) == 2
    assert {r["high_sensor"] for r in groups} == {"VN100", "ADIS16488"}
    for i, group in enumerate(groups):
        error = (0.01, 1.0)[i]
        assert group["samples"] == 100
        assert group["signal"]["virtual"]["gyro_vector_rmse_radps"] == pytest.approx(
            np.sqrt(3) * error / 2
        )
