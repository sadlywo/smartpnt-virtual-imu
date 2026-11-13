from __future__ import annotations

import pandas as pd

from virtual_imu.fir_arx_research.aggregate import _minimal_comparison, _tda_comparison, _vmd_comparison


def test_vmd_comparison_uses_matching_raw_control() -> None:
    frame = pd.DataFrame(
        {
            "protocol": ["p", "p", "p"],
            "condition": ["raw", "raw", "vmd_joint"],
            "model": ["fir", "arx", "fir"],
            "feature_count": [6, 6, 12],
            "test_gyro_vector_rmse": [1.0, 2.0, 0.9],
            "test_accel_vector_rmse": [2.0, 4.0, 2.1],
        }
    )
    result = _vmd_comparison(frame)
    fir = result[result["model"].eq("fir")].iloc[0]
    assert fir["gyro_delta_pct"] < 0
    assert fir["accel_delta_pct"] > 0


def test_tda_comparison_uses_same_windowed_support() -> None:
    frame = pd.DataFrame(
        {
            "protocol": ["p", "p"],
            "pipeline": ["raw_windowed", "tda_routed"],
            "model": ["fir", "fir"],
            "test_gyro_vector_rmse": [1.0, 0.8],
            "test_accel_vector_rmse": [2.0, 2.0],
        }
    )
    result = _tda_comparison(frame)
    assert abs(result.iloc[0]["gyro_delta_pct"] + 20.0) < 1e-12
    assert abs(result.iloc[0]["accel_delta_pct"]) < 1e-12


def test_minimal_comparison_reports_support() -> None:
    metrics = pd.DataFrame(
        {
            "protocol": ["p", "p"],
            "model": ["raw_arx", "minimal_fir_arx"],
            "test_gyro_vector_rmse": [1.0, 0.99],
            "test_accel_vector_rmse": [2.0, 1.98],
        }
    )
    selection = pd.DataFrame(
        {"protocol": ["p", "p"], "component_count": [12, 16], "fallback": [False, True]}
    )
    result = _minimal_comparison(metrics, selection)
    assert result.iloc[0]["mean_component_count"] == 14.0
    assert result.iloc[0]["fallback_fraction"] == 0.5
