from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.eval.supervised_baseline import evaluate_supervised_baselines


def _paired_frame(scale: float, bias: float, sequence_id: str) -> pd.DataFrame:
    low = np.arange(36, dtype=float).reshape(6, 6)
    high = scale * low + bias
    return pd.DataFrame(
        {
            "timestamp_s": np.arange(len(low), dtype=float) * 0.025,
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
            "sequence_id": sequence_id,
        }
    )


def test_supervised_baselines_fit_affine_mapping():
    train = pd.concat(
        [_paired_frame(scale=2.0, bias=1.0, sequence_id="a"), _paired_frame(scale=2.0, bias=1.0, sequence_id="b")],
        ignore_index=True,
    )
    test = _paired_frame(scale=2.0, bias=1.0, sequence_id="c")

    result = evaluate_supervised_baselines(train, test, fir_taps=2, ridge_alpha=1e-8)

    assert result["sample_count"] == len(test)
    assert "identity" in result["baselines"]
    assert "affine" in result["baselines"]
    assert result["baselines"]["affine"]["rmse_6d"] < result["baselines"]["identity"]["rmse_6d"]
    assert result["baselines"]["learned_fir_causal"]["rmse_6d"] < result["baselines"]["identity"]["rmse_6d"]
    assert result["leaderboard"][0]["rmse_6d"] <= result["leaderboard"][-1]["rmse_6d"]
