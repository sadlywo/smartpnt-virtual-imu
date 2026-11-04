from __future__ import annotations

import pandas as pd

from virtual_imu.io.alimu_context import CONTEXT_FEATURE_COLUMNS, add_context_features


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_s": [0.0, 0.1, 0.2, 0.3],
            "gx_low": [0.1, 0.2, 1.0, 1.1],
            "gy_low": [0.0, 0.0, 2.0, 2.1],
            "gz_low": [0.0, 0.0, 3.0, 3.1],
            "ax_low": [0.0, 0.0, 4.0, 4.1],
            "ay_low": [0.0, 0.0, 5.0, 5.1],
            "az_low": [-9.8, -9.7, 6.0, 6.1],
            "gx_high": [10.0, 10.0, 10.0, 10.0],
            "gy_high": [10.0, 10.0, 10.0, 10.0],
            "gz_high": [10.0, 10.0, 10.0, 10.0],
            "ax_high": [10.0, 10.0, 10.0, 10.0],
            "ay_high": [10.0, 10.0, 10.0, 10.0],
            "az_high": [10.0, 10.0, 10.0, 10.0],
            "temperature_c": [25.0, 25.0, 26.0, 26.0],
            "motion_label": ["stationary", "stationary", "dynamic", "dynamic"],
            "sequence_id": ["s1", "s1", "s2", "s2"],
            "low_sensor": ["BMX160", "BMX160", "MPU9250", "MPU9250"],
        }
    )


def test_context_features_use_specs_and_low_only_static_estimates():
    specs = {
        "BMX160": {"gyro_noise_density": 0.014, "accel_noise_density": 0.0018},
        "MPU9250": {"gyro_noise_density": 0.01, "accel_noise_density": 0.003},
    }

    out = add_context_features(_frame(), specs)
    changed_high = _frame()
    for column in [col for col in changed_high.columns if col.endswith("_high")]:
        changed_high[column] += 123.0
    out_changed_high = add_context_features(changed_high, specs)

    assert out["spec_gyro_noise_density"].tolist() == [0.014, 0.014, 0.01, 0.01]
    assert out.loc[2, "static_low_mean_gx"] == 1.05
    pd.testing.assert_frame_equal(out[CONTEXT_FEATURE_COLUMNS], out_changed_high[CONTEXT_FEATURE_COLUMNS])
