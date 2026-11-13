from __future__ import annotations

from pathlib import Path

import pandas as pd

from virtual_imu.fir_arx_research.branch_viz import plot_branch_effects, plot_branch_summary


def _comparison() -> pd.DataFrame:
    rows = []
    for index, protocol in enumerate(["alimu_only", "penn_only"]):
        for branch, condition, model in (
            ("minimal", "scenario_minimal", "fir_arx"),
            ("tda", "tda_routed", "fir"),
        ):
            rows.append(
                {
                    "branch": branch,
                    "protocol": protocol,
                    "condition": condition,
                    "model": model,
                    "gyro_delta_pct": -1.0 + index,
                    "accel_delta_pct": -0.8 + index,
                }
            )
    return pd.DataFrame(rows)


def test_branch_figures_export_all_formats(tmp_path) -> None:
    comparison = _comparison()
    summary = pd.DataFrame(
        {
            "branch": ["minimal", "tda"],
            "condition": ["scenario_minimal", "tda_routed"],
            "model": ["fir_arx", "fir"],
            "protocols": [2, 2],
            "gyro_delta_pct_mean": [-1.0, 0.0],
            "gyro_delta_pct_median": [-1.0, 0.0],
            "accel_delta_pct_mean": [-0.8, 0.0],
            "accel_delta_pct_median": [-0.8, 0.0],
            "joint_score_delta_mean": [-0.9, 0.0],
            "joint_score_delta_median": [-0.9, 0.0],
            "branch_win_protocols": [2, 0],
        }
    )
    first = plot_branch_effects(comparison, tmp_path)
    second = plot_branch_summary(summary, tmp_path)
    for record in (first, second):
        for suffix in ("svg", "pdf", "png", "tiff"):
            assert Path(record["files"][suffix]).is_file()
