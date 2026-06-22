from __future__ import annotations

import numpy as np
import pandas as pd

from virtual_imu.eval.allan import allan_deviation
from virtual_imu.eval.baselines import evaluate_baselines
from virtual_imu.eval.reports import evaluate_predictions
from virtual_imu.eval.signal_metrics import signal_metrics


def _prediction_frame(n: int = 200) -> pd.DataFrame:
    t = np.arange(n) * 0.01
    high_g = np.column_stack([0.1 * np.sin(t), 0.05 * np.cos(t), 0.02 * np.sin(2 * t)])
    high_a = np.column_stack([0.1 * np.sin(t), 0.05 * np.cos(t), 9.80665 + 0.02 * np.sin(2 * t)])
    df = pd.DataFrame({"timestamp_s": t, "sequence_id": "m"})
    for i, axis in enumerate(["gx", "gy", "gz"]):
        df[f"{axis}_high"] = high_g[:, i]
        df[f"{axis}_low"] = high_g[:, i] + 0.01
        df[f"{axis}_virtual"] = high_g[:, i] + 0.002
    for i, axis in enumerate(["ax", "ay", "az"]):
        df[f"{axis}_high"] = high_a[:, i]
        df[f"{axis}_low"] = high_a[:, i] + 0.05
        df[f"{axis}_virtual"] = high_a[:, i] + 0.01
    return df


def test_signal_metrics_improve_for_virtual():
    df = _prediction_frame()
    low = signal_metrics(df, "low")
    virtual = signal_metrics(df, "virtual")
    assert virtual["gyro_norm_rmse"] < low["gyro_norm_rmse"]
    assert virtual["accel_norm_rmse"] < low["accel_norm_rmse"]


def test_allan_and_evaluation_outputs(tmp_path):
    df = _prediction_frame()
    taus, adev = allan_deviation(df["gx_high"].to_numpy(), rate_hz=100.0)
    assert len(taus) == len(adev)
    pred_path = tmp_path / "pred.parquet"
    out_path = tmp_path / "metrics.json"
    df.to_parquet(pred_path, index=False)
    metrics = evaluate_predictions(pred_path, out_path, min_allan_duration_s=20.0)
    assert out_path.exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "metrics.md").exists()
    assert metrics["allan"]["high"]["status"] == "insufficient duration"


def test_baseline_evaluation_outputs(tmp_path):
    df = _prediction_frame()
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    out_path = tmp_path / "baseline_metrics.json"
    df.to_parquet(train_path, index=False)
    df.to_parquet(test_path, index=False)
    result = evaluate_baselines(
        test_path,
        out_path,
        train_path=train_path,
        predictions_dir=tmp_path / "baseline_predictions",
        min_allan_duration_s=20.0,
        neural_epochs=1,
        neural_window_size=32,
        neural_stride=16,
    )
    assert out_path.exists()
    assert (tmp_path / "baseline_metrics.csv").exists()
    assert (tmp_path / "baseline_metrics.md").exists()
    assert "identity" in result["baselines"]
    assert "ronin_resnet_adapter" in result["baselines"]
    assert (tmp_path / "baseline_predictions" / "affine.parquet").exists()
    assert (tmp_path / "figures" / "baseline_summary_grid.png").exists()
