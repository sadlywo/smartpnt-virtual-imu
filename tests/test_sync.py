from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.preprocessing.sync import align_low_high, estimate_time_delay_s


def _frame(prefix: str, t: np.ndarray, delay: float = 0.0) -> pd.DataFrame:
    s = np.sin(2 * np.pi * 1.0 * (t - delay))
    return pd.DataFrame(
        {
            "timestamp_s": t,
            f"gx_{prefix}": s,
            f"gy_{prefix}": 0.2 * s,
            f"gz_{prefix}": 0.1 * s,
            f"ax_{prefix}": 0.0,
            f"ay_{prefix}": 0.0,
            f"az_{prefix}": 9.80665,
        }
    )


def test_estimate_time_delay_and_align():
    t = np.arange(0, 5, 0.01)
    high = _frame("high", t, delay=0.0)
    low = _frame("low", t, delay=0.05)
    lag = estimate_time_delay_s(low, high, rate_hz=100.0, max_lag_s=0.2)
    assert abs(abs(lag) - 0.05) <= 0.02
    aligned, report = align_low_high(low, high, target_rate_hz=100.0, estimate_delay=True, max_lag_s=0.2)
    assert len(aligned) > 100
    assert "estimated_delay_s" in report
