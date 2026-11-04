from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from virtual_imu.preprocessing.windowing import make_window_slices
from virtual_imu.training.dataset import LOW_IMU_COLUMNS, TARGET_COLUMNS, IMUWindowDataset
from virtual_imu.training.trainer import train_model

VIRTUAL_COLUMNS = [column.replace("_high", "_virtual") for column in TARGET_COLUMNS]
TARGET_STATS = {
    "columns": TARGET_COLUMNS,
    "mean": [0.7, -0.3, 0.2, 2.0, -3.0, 0.5],
    "std": [0.1, 0.5, 0.2, 3.0, 2.0, 4.0],
}


def _frame(
    sequence: str,
    n: int = 8,
    dataset: str = "alimu",
    low_sensor: str = "BMX160",
    high_sensor: str = "VN100",
    gyro_ratio: float = 0.5,
    accel_ratio: float = 0.5,
) -> pd.DataFrame:
    high = np.column_stack([np.arange(n, dtype=float) / 10 + axis for axis in range(6)])
    error = np.tile([0.2, 0.0, 0.0, 5.0, 0.0, 0.0], (n, 1))
    frame = pd.DataFrame(high, columns=TARGET_COLUMNS)
    frame[LOW_IMU_COLUMNS] = high + error
    frame[VIRTUAL_COLUMNS] = high + error * np.array([gyro_ratio] * 3 + [accel_ratio] * 3)
    frame["sequence_id"] = sequence
    frame["timestamp_s"] = np.arange(n) / 40
    frame["dataset_id"] = dataset
    frame["low_sensor"] = low_sensor
    frame["high_sensor"] = high_sensor
    return frame


def _prediction_batches(
    frame: pd.DataFrame,
    *,
    window_size: int = 4,
    stride: int = 2,
    target_mode: str = "direct",
) -> list[np.ndarray]:
    windows = []
    for _, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values("timestamp_s").reset_index(drop=True)
        values = group[VIRTUAL_COLUMNS].to_numpy(float)
        if target_mode == "residual":
            values -= group[LOW_IMU_COLUMNS].to_numpy(float)
        values = (values - np.asarray(TARGET_STATS["mean"])) / np.asarray(TARGET_STATS["std"])
        windows.extend(values[slc].T for slc in make_window_slices(len(group), window_size, stride))
    return [np.stack(windows)] if windows else []


def _score(frame, batches, *, window_size=4, stride=2, target_mode="direct", stats=None):
    from virtual_imu.training.paired_validation import grouped_relative_rmse_validation

    return grouped_relative_rmse_validation(
        frame,
        batches,
        TARGET_STATS if stats is None else stats,
        window_size=window_size,
        stride=stride,
        target_mode=target_mode,
    )


@pytest.mark.parametrize("target_mode", ["direct", "residual"])
def test_dataset_and_pair_means_are_equal_weighted_in_physical_units(target_mode):
    frame = pd.concat(
        [
            _frame("bmx", 40, low_sensor="BMX160", gyro_ratio=0.0, accel_ratio=0.4),
            _frame("mpu", 8, low_sensor="MPU9250", gyro_ratio=0.2, accel_ratio=0.6),
            _frame("lsm", 4, low_sensor="LSM9DS1", gyro_ratio=0.4, accel_ratio=0.8),
            _frame("penn", 6, "penncosyvio", "Tango", "ADIS16488", 0.6, 1.0),
        ],
        ignore_index=True,
    )
    metrics, details = _score(frame, _prediction_batches(frame, target_mode=target_mode), target_mode=target_mode)

    assert metrics["val_grouped_score"] == pytest.approx(0.6)
    assert metrics["val_score"] == metrics["val_grouped_score"]
    assert metrics["val_relative_gyro_rmse"] == pytest.approx(0.4)
    assert metrics["val_relative_accel_rmse"] == pytest.approx(0.8)
    assert metrics["val_pair_count"] == 4
    assert metrics["val_dataset_count"] == 2
    assert [pair["score"] for pair in details["pairs"]] == pytest.approx([0.2, 0.4, 0.6, 0.8])
    assert [dataset["score"] for dataset in details["datasets"]] == pytest.approx([0.4, 0.8])
    assert details["baseline_epsilon"] == 1e-12
    json.dumps(details, allow_nan=False)


def test_pair_rmse_pools_unique_timestamps_across_sequences_before_taking_root():
    frame = pd.concat([_frame("short", 4, gyro_ratio=1, accel_ratio=1),
                       _frame("long", 8, gyro_ratio=3, accel_ratio=3)], ignore_index=True)
    metrics, details = _score(frame, _prediction_batches(frame))

    expected = np.sqrt((4 * 1**2 + 8 * 3**2) / 12)
    assert metrics["val_score"] == pytest.approx(expected)
    assert details["pairs"][0]["samples"] == 12
    assert details["pairs"][0]["sequence_count"] == 2


@pytest.mark.parametrize("target_mode", ["direct", "residual"])
def test_overlaps_are_averaged_and_uncovered_tail_is_excluded(target_mode, monkeypatch):
    frame = _frame("z-first", 7)
    frame[TARGET_COLUMNS] = 0.0
    frame[LOW_IMU_COLUMNS] = [2.0, 0.0, 0.0, 4.0, 0.0, 0.0]
    frame.loc[6, LOW_IMU_COLUMNS] = 1e9
    predictions = np.zeros((2, 4, 6))
    predictions[0, 2:, [0, 3]] = 2.0
    predictions[1, :2, [0, 3]] = -2.0
    if target_mode == "residual":
        predictions -= frame.loc[:3, LOW_IMU_COLUMNS].to_numpy(float)
    normalized = (predictions - TARGET_STATS["mean"]) / TARGET_STATS["std"]
    batches = [normalized[:1].transpose(0, 2, 1), normalized[1:].transpose(0, 2, 1)]
    shuffled = frame.sample(frac=1.0, random_state=9)

    def reject_temporal_join(*args, **kwargs):
        raise AssertionError("validation used a temporal diagnostic on pooled sequences/windows")

    monkeypatch.setattr(np.fft, "rfft", reject_temporal_join)
    monkeypatch.setattr(np, "cumsum", reject_temporal_join)
    metrics, details = _score(shuffled, batches, target_mode=target_mode)

    assert metrics["val_score"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["val_covered_samples"] == 6
    assert metrics["val_uncovered_samples"] == 1
    assert metrics["val_identity_gyro_rmse"] == pytest.approx(2.0)
    assert metrics["val_identity_accel_rmse"] == pytest.approx(4.0)
    assert details["sequences"][0]["max_overlap"] == 2
    assert details["sequences"][0]["window_count"] == 2


def test_window_order_matches_dataset_and_skips_short_sequences():
    frame = pd.concat([_frame("z", 8), _frame("short", 3), _frame("a", 6, gyro_ratio=0.1, accel_ratio=0.9)], ignore_index=True)
    frame = frame.groupby("sequence_id", sort=False, group_keys=False).sample(frac=1.0, random_state=1)
    dataset = IMUWindowDataset(frame, 4, 2, target_stats=TARGET_STATS)
    batches = _prediction_batches(frame)
    assert sum(len(batch) for batch in batches) == len(dataset)
    metrics, details = _score(frame, batches)

    expected = (np.sqrt((8 * 0.5**2 + 6 * 0.1**2) / 14)
                + np.sqrt((8 * 0.5**2 + 6 * 0.9**2) / 14)) / 2
    assert metrics["val_score"] == pytest.approx(expected)
    assert metrics["val_covered_samples"] == 14
    assert metrics["val_uncovered_samples"] == 3
    assert [row["sequence_id"] for row in details["sequences"]] == ["z", "short", "a"]
    assert details["sequences"][1]["status"] == "no_complete_windows"


def test_entirely_uncovered_pair_cannot_silently_disappear_from_balanced_score():
    frame = pd.concat([_frame("covered", 4), _frame("short", 3, low_sensor="MPU9250")], ignore_index=True)
    with pytest.raises(ValueError, match=r"no covered.*MPU9250"):
        _score(frame, _prediction_batches(frame))


def test_stride_gaps_count_only_actual_covered_timestamps():
    frame = _frame("gaps", 10)
    frame.loc[[2, 3, 6, 7], LOW_IMU_COLUMNS] = 1e9
    metrics, details = _score(frame, _prediction_batches(frame, window_size=2, stride=4), window_size=2, stride=4)

    assert metrics["val_score"] == pytest.approx(0.5)
    assert metrics["val_covered_samples"] == 6
    assert metrics["val_uncovered_samples"] == 4
    assert details["sequences"][0]["max_overlap"] == 1


def test_residual_reconstruction_preserves_original_si_low_and_high_precision():
    frame = _frame("precision", 4)
    high = 1e8 + np.arange(24).reshape(4, 6)
    error = np.tile([0.125, 0.0, 0.0, 0.5, 0.0, 0.0], (4, 1))
    frame[TARGET_COLUMNS] = high
    frame[LOW_IMU_COLUMNS] = high + error
    frame[VIRTUAL_COLUMNS] = high + error / 2
    metrics, _ = _score(frame, _prediction_batches(frame, target_mode="residual"), target_mode="residual")

    assert metrics["val_score"] == pytest.approx(0.5)
    assert metrics["val_gyro_rmse"] == pytest.approx(0.0625)
    assert metrics["val_accel_rmse"] == pytest.approx(0.25)


def test_near_zero_identity_denominators_use_declared_floor_and_flags():
    frame = _frame("zero", 4)
    frame[TARGET_COLUMNS] = 0.0
    frame[LOW_IMU_COLUMNS] = [0.0, 0.0, 0.0, 1e-14, 0.0, 0.0]
    frame[VIRTUAL_COLUMNS] = [2e-12, 0.0, 0.0, 3e-12, 0.0, 0.0]
    stats = {"columns": TARGET_COLUMNS, "mean": [0.0] * 6, "std": [1.0] * 6}
    batches = [frame[VIRTUAL_COLUMNS].to_numpy(float).T[None]]
    metrics, details = _score(frame, batches, stats=stats)

    assert metrics["val_score"] == pytest.approx(2.5)
    assert metrics["val_near_zero_baseline_pairs"] == 1
    pair = details["pairs"][0]
    assert pair["gyro_baseline_near_zero"] is True
    assert pair["accel_baseline_near_zero"] is True
    assert pair["identity_gyro_rmse_radps"] == 0.0
    assert pair["identity_accel_rmse_mps2"] == pytest.approx(1e-14)
    json.dumps(details, allow_nan=False)


@pytest.mark.parametrize("batches, message", [
    ([], "window count"),
    ([np.zeros((2, 6, 4))], "window count"),
    ([np.zeros((1, 4, 6))], "shape"),
    ([np.full((1, 6, 4), np.nan)], "non-finite"),
])
def test_prediction_window_mismatches_fail_clearly(batches, message):
    with pytest.raises(ValueError, match=message):
        _score(_frame("one", 4), batches)


@pytest.mark.parametrize("column", ["dataset_id", "low_sensor", "high_sensor", "sequence_id", "gx_high", "gx_low"])
def test_missing_source_columns_fail_clearly(column):
    frame = _frame("one", 4)
    batches = _prediction_batches(frame)
    with pytest.raises(ValueError, match=column):
        _score(frame.drop(columns=column), batches)


def test_sequence_must_have_one_pair_and_unique_finite_timestamps():
    frame = _frame("one", 4)
    batches = _prediction_batches(frame)
    frame.loc[1, "low_sensor"] = "another"
    with pytest.raises(ValueError, match=r"one.*low_sensor"):
        _score(frame, batches)
    frame.loc[1, "low_sensor"] = frame.loc[0, "low_sensor"]
    frame.loc[1, "timestamp_s"] = frame.loc[0, "timestamp_s"]
    with pytest.raises(ValueError, match="timestamp"):
        _score(frame, batches)
    frame.loc[1, "timestamp_s"] = np.nan
    with pytest.raises(ValueError, match="timestamp"):
        _score(frame, batches)


def test_unsupported_target_mode_and_invalid_stats_are_rejected():
    frame = _frame("one", 4)
    batches = _prediction_batches(frame)
    with pytest.raises(ValueError, match=r"direct.*residual"):
        _score(frame, batches, target_mode="competitor_residual")
    with pytest.raises(ValueError, match="target_stats"):
        _score(frame, batches, stats={**TARGET_STATS, "std": [0.0] * 6})
    with pytest.raises(ValueError, match="target_stats"):
        _score(frame, batches, stats={**TARGET_STATS, "columns": list(reversed(TARGET_COLUMNS))})


def _training_config(tmp_path, *, metric="val_grouped_score"):
    frame = _frame("one", 12)
    # Symmetric residual labels make the zero normalized output exactly identity.
    signs = np.where(np.arange(len(frame)) % 2, 1.0, -1.0)[:, None]
    frame[TARGET_COLUMNS] = frame[LOW_IMU_COLUMNS].to_numpy(float) + signs * np.array([0.2] * 3 + [5.0] * 3)
    path = tmp_path / "paired.parquet"
    frame.to_parquet(path, index=False)
    return {
        "seed": 17,
        "experiment_name": "grouped_validation",
        "data": {
            "train_path": str(path), "val_path": str(path),
            "feature_columns": LOW_IMU_COLUMNS,
            "window_size": 4, "stride": 2, "batch_size": 3,
        },
        "model": {
            "type": "tcn", "target_mode": "residual", "hidden_channels": 4,
            "levels": 1, "kernel_size": 3, "dropout": 0.0, "zero_init_head": True,
        },
        "training": {
            "epochs": 1, "lr": 0.0, "device": "cpu", "checkpoint_metric": metric,
            "validation_score": {"method": "grouped_relative_rmse"},
        },
        "output": {"runs_dir": str(tmp_path / "runs")},
    }


def test_formal_trainer_bypasses_legacy_diagnostics_and_saves_group_details(tmp_path, monkeypatch):
    def reject_legacy(*args, **kwargs):
        raise AssertionError("formal validation called legacy pooled temporal diagnostics")

    monkeypatch.setattr("virtual_imu.training.trainer._validation_composite_score", reject_legacy)
    run_dir = train_model(_training_config(tmp_path))
    metrics = pd.read_csv(run_dir / "metrics.csv")
    epoch = json.loads((run_dir / "validation" / "epoch_0001.json").read_text(encoding="utf-8"))
    best = json.loads((run_dir / "validation" / "best.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "best.ckpt", map_location="cpu", weights_only=False)

    assert metrics.loc[0, "val_score"] == pytest.approx(1.0, abs=1e-6)
    assert metrics.loc[0, "val_grouped_score"] == metrics.loc[0, "val_score"]
    assert "val_gyro_spectral" not in metrics.columns
    assert epoch == best
    assert best["epoch"] == 1
    assert best["checkpoint_metric"] == "val_grouped_score"
    assert best["pairs"][0]["samples"] == 12
    assert checkpoint["best_metric"] == pytest.approx(metrics.loc[0, "val_grouped_score"])
    assert checkpoint["validation_details_path"] == "validation/best.json"


def test_formal_trainer_rejects_unknown_monitor_instead_of_falling_back_to_loss(tmp_path):
    with pytest.raises(ValueError, match=r"checkpoint_metric.*val_typo"):
        train_model(_training_config(tmp_path, metric="val_typo"))


def test_checkpoint_scheduler_and_early_stopping_share_requested_grouped_score(tmp_path, monkeypatch):
    config = _training_config(tmp_path)
    config["training"].update({
        "epochs": 5, "early_stopping_patience": 2,
        "lr_scheduler": "reduce_on_plateau", "lr": 0.001,
    })
    scores = iter([0.3, 0.4, 0.5, 0.1, 0.0])
    seen = []

    def prescribed_scores(*args, **kwargs):
        score = next(scores)
        return {"val_score": score, "val_grouped_score": score}, {"pairs": [], "score": score}

    original_step = torch.optim.lr_scheduler.ReduceLROnPlateau.step

    def record_step(scheduler, metrics, epoch=None):
        seen.append(metrics)
        return original_step(scheduler, metrics, epoch)

    monkeypatch.setattr("virtual_imu.training.trainer.grouped_relative_rmse_validation", prescribed_scores, raising=False)
    monkeypatch.setattr(torch.optim.lr_scheduler.ReduceLROnPlateau, "step", record_step)
    run_dir = train_model(config)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    checkpoint = torch.load(run_dir / "best.ckpt", map_location="cpu", weights_only=False)
    best = json.loads((run_dir / "validation" / "best.json").read_text(encoding="utf-8"))

    assert seen == [0.3, 0.4, 0.5]
    assert len(metrics) == 3
    assert checkpoint["best_epoch"] == 1
    assert checkpoint["best_metric"] == pytest.approx(0.3)
    assert best["epoch"] == 1
    assert len(list((run_dir / "validation").glob("epoch_*.json"))) == 3
