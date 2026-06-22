from __future__ import annotations

import pandas as pd

from virtual_imu.preprocessing.windowing import make_window_slices, window_long_dataframe


def test_make_window_slices():
    slices = make_window_slices(10, window_size=4, stride=3)
    assert [(s.start, s.stop) for s in slices] == [(0, 4), (3, 7), (6, 10)]


def test_window_long_dataframe():
    df = pd.DataFrame({"timestamp_s": range(6), "sequence_id": ["a"] * 6, "gx_low": range(6)})
    out = window_long_dataframe(df, window_size=3, stride=2)
    assert out["window_id"].nunique() == 2
    assert set(out["sample_in_window"]) == {0, 1, 2}
