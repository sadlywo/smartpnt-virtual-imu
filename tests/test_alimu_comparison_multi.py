from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.eval.comparison_table import build_comparison_table


def _prediction_frame(high: np.ndarray, low: np.ndarray, offset: float) -> pd.DataFrame:
    virtual = high + offset
    return pd.DataFrame(
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


def test_build_comparison_table_accepts_multiple_neural_predictions():
    supervised = {
        "leaderboard": [
            {
                "baseline": "identity",
                "rmse_6d": 2.0,
                "gyro_rmse_3d": 1.0,
                "accel_rmse_3d": 1.5,
                "mae_6d": 0.9,
            }
        ]
    }
    high = np.ones((3, 6))
    low = high + 1.0

    table = build_comparison_table(
        supervised,
        {
            "TCN": _prediction_frame(high, low, 0.5),
            "ALiMU-inspired": _prediction_frame(high, low, 0.1),
        },
    )

    assert table["method"].tolist() == ["ALiMU-inspired", "TCN", "identity"]
    assert table["rank"].tolist() == [1, 2, 3]
