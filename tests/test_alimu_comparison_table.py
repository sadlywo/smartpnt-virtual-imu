from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.eval.comparison_table import build_comparison_table


def test_build_comparison_table_adds_tcn_row_and_sorts_by_rmse():
    supervised = {
        "leaderboard": [
            {
                "baseline": "identity",
                "rmse_6d": 2.0,
                "gyro_rmse_3d": 1.0,
                "accel_rmse_3d": 1.5,
                "mae_6d": 0.9,
            },
            {
                "baseline": "affine",
                "rmse_6d": 1.2,
                "gyro_rmse_3d": 0.7,
                "accel_rmse_3d": 1.0,
                "mae_6d": 0.5,
            },
        ]
    }
    high = np.array([[1.0, 2.0, 3.0, 9.0, 8.0, 7.0], [2.0, 3.0, 4.0, 8.0, 7.0, 6.0]])
    low = high + 1.0
    virtual = high + 0.25
    pred = pd.DataFrame(
        {
            "gx_low": low[:, 0],
            "gy_low": low[:, 1],
            "gz_low": low[:, 2],
            "ax_low": low[:, 3],
            "ay_low": low[:, 4],
            "az_low": low[:, 5],
            "gx_high": high[:, 0],
            "gy_high": high[:, 1],
            "gz_high": high[:, 2],
            "ax_high": high[:, 3],
            "ay_high": high[:, 4],
            "az_high": high[:, 5],
            "gx_virtual": virtual[:, 0],
            "gy_virtual": virtual[:, 1],
            "gz_virtual": virtual[:, 2],
            "ax_virtual": virtual[:, 3],
            "ay_virtual": virtual[:, 4],
            "az_virtual": virtual[:, 5],
        }
    )

    table = build_comparison_table(supervised, pred, neural_name="TCN full")

    assert table.iloc[0]["method"] == "TCN full"
    assert table.iloc[0]["rmse_6d"] < table.loc[table["method"] == "identity", "rmse_6d"].iloc[0]
    assert table.iloc[0]["gyro_snr_improvement_db"] > 0
    assert table["rank"].tolist() == [1, 2, 3]
