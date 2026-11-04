from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from virtual_imu.eval.alimu_matrix_aggregation import (
    ALIMU_ANALYTIC_METHODS,
    CORE_METRICS,
    aggregate_alimu_trials,
    alimu_trial_row,
    write_alimu_matrix_aggregation,
)
from virtual_imu.eval.alimu_matrix_report import (
    render_alimu_matrix_report,
    write_alimu_matrix_report,
)
from virtual_imu.training.alimu_matrix_config import ALIMU_NEURAL_METHODS, ALIMU_SEEDS


def _metrics(value: float, sample_count: int = 40) -> dict[str, object]:
    return {
        "sample_count": sample_count,
        "summary": {
            "gyro_rmse": value,
            "gyro_mae": value + 0.01,
            "accel_rmse": value + 0.02,
            "accel_mae": value + 0.03,
            "mean_axis_correlation": 0.9 - value / 100.0,
        },
        "spectral": {
            "method": {
                "gyro_frequency_band_rmse": value + 0.04,
                "accel_frequency_band_rmse": value + 0.05,
                "gyro_coherence": 0.8,
                "accel_coherence": 0.7,
                "gyro_phase_delay_s": value / 100.0,
                "accel_phase_delay_s": value / 200.0,
            }
        },
        "preintegration": {
            "method": {
                "imu_preintegration_residual": value + 0.06,
                "delta_rotation_error_rad": value + 0.07,
                "delta_velocity_error_m_s": value + 0.08,
                "delta_position_proxy_error_m": value + 0.09,
            }
        },
    }


def _complete_trials() -> tuple[pd.DataFrame, pd.DataFrame]:
    neural = []
    for method_index, method in enumerate(ALIMU_NEURAL_METHODS):
        for seed_index, seed in enumerate(ALIMU_SEEDS):
            neural.append(
                alimu_trial_row("MPU9250", method, seed, _metrics(0.5 + method_index + seed_index))
            )
    analytic = [
        alimu_trial_row("MPU9250", method, None, _metrics(2.0 + index))
        for index, method in enumerate(ALIMU_ANALYTIC_METHODS)
    ]
    return pd.DataFrame(neural), pd.DataFrame(analytic)


def _provenance() -> dict[str, object]:
    run_id = "alimu_leave_sensor_mpu9250_20260713"
    run_dir = "runs/alimu_leave_sensor_mpu9250_20260713"
    return {
        "git_commit": "abc1234",
        "run_id": run_id,
        "run_dir": run_dir,
        "config_snapshots": {
            "matrix": {"path": "runs/alimu/matrix.yaml", "sha256": "a" * 64},
            "base": {"path": "runs/alimu/base.yaml", "sha256": "b" * 64},
        },
        "split_manifest": {
            "path": "data/processed/alimu_strict/leave_sensor_mpu9250/manifest.json",
            "sha256": "c" * 64,
        },
        "inputs": {
            "train": {"path": "data/processed/alimu_strict/leave_sensor_mpu9250/train.parquet", "sha256": "d" * 64},
            "val": {"path": "data/processed/alimu_strict/leave_sensor_mpu9250/val.parquet", "sha256": "e" * 64},
            "test": {"path": "data/processed/alimu_strict/leave_sensor_mpu9250/test.parquet", "sha256": "f" * 64},
        },
        "artifacts": {
            "trials": {"path": f"{run_dir}/trials.csv", "run_id": run_id},
            "overall_summary": {"path": f"{run_dir}/overall_summary.csv", "run_id": run_id},
        },
        "split_provenance": {
            "mode": "leave_sensor_out",
            "holdout_sensor": "MPU9250",
            "train_sensors": ["BMX160", "LSM9DS1"],
            "validation_sensors": ["BMX160", "LSM9DS1"],
            "test_sensors": ["MPU9250"],
            "target_sensor": "VN100",
            "sample_rate_hz": 40,
        },
    }


def _reverse_mapping_order(value: object) -> object:
    if isinstance(value, dict):
        return {key: _reverse_mapping_order(item) for key, item in reversed(value.items())}
    if isinstance(value, list):
        return [_reverse_mapping_order(item) for item in value]
    return value


def test_trial_row_flattens_canonical_payload_with_holdout_sensor() -> None:
    row = alimu_trial_row("MPU9250", "tcn", 20260712, _metrics(1.0))

    assert row["holdout_sensor"] == "MPU9250"
    assert row["normalized_rmse_score"] != row["normalized_rmse_score"]
    assert set(CORE_METRICS) <= set(row)


def test_aggregate_requires_exact_fifteen_and_four_trial_axes() -> None:
    neural, analytic = _complete_trials()
    summary = aggregate_alimu_trials(neural, analytic)

    assert len(neural) == 15
    assert len(analytic) == 4
    assert len(summary) == 9
    assert set(summary["method"]) == set(ALIMU_NEURAL_METHODS) | set(ALIMU_ANALYTIC_METHODS)


@pytest.mark.parametrize("mutation, match", [
    (lambda neural, analytic: pd.concat([neural, neural.iloc[[0]]], ignore_index=True), "duplicate"),
    (lambda neural, analytic: neural.iloc[1:].copy(), "complete 15-row"),
])
def test_aggregate_rejects_duplicate_or_missing_neural_axes(mutation, match: str) -> None:
    neural, analytic = _complete_trials()

    with pytest.raises(ValueError, match=match):
        aggregate_alimu_trials(mutation(neural, analytic), analytic)


def test_aggregate_rejects_invalid_neural_seed_and_analytic_seed() -> None:
    neural, analytic = _complete_trials()
    neural.loc[0, "seed"] = 1
    with pytest.raises(ValueError, match="configured integers"):
        aggregate_alimu_trials(neural, analytic)

    neural, analytic = _complete_trials()
    analytic.loc[0, "seed"] = ALIMU_SEEDS[0]
    with pytest.raises(ValueError, match="analytic_trials seed values must be null"):
        aggregate_alimu_trials(neural, analytic)


@pytest.mark.parametrize("column, value", [("gyro_rmse", np.inf), ("sample_count", 0)])
def test_aggregate_rejects_nonfinite_or_invalid_values(column: str, value: float) -> None:
    neural, analytic = _complete_trials()
    analytic.loc[0, column] = value

    with pytest.raises(ValueError, match=r"non-finite|positive"):
        aggregate_alimu_trials(neural, analytic)


def test_aggregate_requires_one_common_holdout_and_sample_count() -> None:
    neural, analytic = _complete_trials()
    neural.loc[0, "holdout_sensor"] = "other"
    with pytest.raises(ValueError, match="holdout_sensor"):
        aggregate_alimu_trials(neural, analytic)

    neural, analytic = _complete_trials()
    analytic.loc[0, "sample_count"] = 41
    with pytest.raises(ValueError, match="sample_count"):
        aggregate_alimu_trials(neural, analytic)


def test_aggregate_scores_seed_means_stds_rank_and_improvements() -> None:
    neural, analytic = _complete_trials()
    summary = aggregate_alimu_trials(neural.sample(frac=1.0, random_state=1), analytic.sample(frac=1.0, random_state=2))

    tcn = summary.loc[summary["method"].eq("tcn")].iloc[0]
    identity = summary.loc[summary["method"].eq("identity")].iloc[0]
    assert tcn["gyro_rmse_mean"] == pytest.approx(1.5)
    assert tcn["gyro_rmse_seed_std"] == pytest.approx(1.0)
    assert tcn["normalized_rmse_score_mean"] == pytest.approx(0.7512376237623762)
    assert identity["normalized_rmse_score_mean"] == pytest.approx(1.0)
    assert identity["gyro_rmse_seed_std"] != identity["gyro_rmse_seed_std"]
    assert summary["rank"].tolist() == list(range(1, 10))
    assert tcn["normalized_rmse_improvement_vs_identity_pct"] == pytest.approx(24.876237623762375)
    assert "fold_count" not in summary and "worst_fold" not in summary
    assert not any("fold_" in column for column in summary.columns)


def test_aggregation_and_artifacts_are_deterministic_and_have_no_fold_table(tmp_path: Path) -> None:
    neural, analytic = _complete_trials()
    trials = pd.concat([neural, analytic], ignore_index=True)
    first = aggregate_alimu_trials(neural, analytic)
    second = aggregate_alimu_trials(neural.iloc[::-1], analytic.iloc[::-1])
    pd.testing.assert_frame_equal(first, second)

    paths = write_alimu_matrix_aggregation(trials, first, tmp_path)
    assert set(paths) == {"trials", "overall_summary"}
    for formats in paths.values():
        assert set(formats) == {"csv", "json", "markdown"}
        assert all(path.exists() for path in formats.values())
        assert json.loads(formats["json"].read_text(encoding="utf-8"))

    shuffled_paths = write_alimu_matrix_aggregation(
        trials.sample(frac=1.0, random_state=5),
        second.sample(frac=1.0, random_state=6),
        tmp_path / "shuffled",
    )
    for name in paths:
        for format_name in paths[name]:
            assert paths[name][format_name].read_bytes() == shuffled_paths[name][format_name].read_bytes()


@pytest.mark.parametrize("field", ["fold", "fold_count", "worst_fold", "gyro_rmse_fold_std", "fold_summary"])
def test_writer_rejects_fold_fields(field: str, tmp_path: Path) -> None:
    neural, analytic = _complete_trials()
    trials = pd.concat([neural, analytic], ignore_index=True)
    summary = aggregate_alimu_trials(neural, analytic)
    summary[field] = 1

    with pytest.raises(ValueError, match="fold"):
        write_alimu_matrix_aggregation(trials, summary, tmp_path)


def test_writer_rejects_forged_summary(tmp_path: Path) -> None:
    neural, analytic = _complete_trials()
    trials = pd.concat([neural, analytic], ignore_index=True)
    summary = aggregate_alimu_trials(neural, analytic)
    summary.loc[0, "gyro_rmse_mean"] = 999.0

    with pytest.raises(ValueError, match="canonical aggregate"):
        write_alimu_matrix_aggregation(trials, summary, tmp_path)


def test_report_is_deterministic_and_states_alimu_contract(tmp_path: Path) -> None:
    neural, analytic = _complete_trials()
    trials = pd.concat([neural, analytic], ignore_index=True)
    summary = aggregate_alimu_trials(neural, analytic)
    report = render_alimu_matrix_report(summary.sample(frac=1.0, random_state=3), trials.sample(frac=1.0, random_state=4), _provenance())

    for claim in (
        "严格 leave_sensor_out",
        "BMX160、LSM9DS1",
        "MPU9250",
        "VN100",
        "40 Hz",
        "15 个神经网络试验",
        "4 个解析基线",
        "均值 ± seed 标准差",
        "不计算 fold 标准差",
        "四分支消融",
        "FIR",
        "bias_drift",
        "种子稳定性",
        "频谱",
        "预积分",
        "不可直接比较",
    ):
        assert claim in report
    assert "fold_count" not in report
    assert "worst_fold" not in report
    assert "_fold_std" not in report

    provenance = _provenance()
    run_dir = tmp_path / "run"
    provenance["run_dir"] = str(run_dir)
    provenance["artifacts"]["trials"]["path"] = str(run_dir / "trials.csv")
    provenance["artifacts"]["overall_summary"]["path"] = str(run_dir / "overall_summary.csv")
    path = write_alimu_matrix_report(summary, trials, provenance, run_dir / "report.md")
    assert path.read_text(encoding="utf-8") == render_alimu_matrix_report(summary, trials, provenance)


def test_report_writer_rejects_output_path_outside_provenance_run_dir(tmp_path: Path) -> None:
    neural, analytic = _complete_trials()
    trials = pd.concat([neural, analytic], ignore_index=True)
    summary = aggregate_alimu_trials(neural, analytic)
    provenance = _provenance()
    run_dir = tmp_path / "run"
    provenance["run_dir"] = str(run_dir)
    provenance["artifacts"]["trials"]["path"] = str(run_dir / "trials.csv")
    provenance["artifacts"]["overall_summary"]["path"] = str(run_dir / "overall_summary.csv")

    with pytest.raises(ValueError, match=r"output_path must be under provenance\.run_dir"):
        write_alimu_matrix_report(summary, trials, provenance, run_dir / ".." / "report.md")


def test_report_rejects_forged_summary_before_rendering() -> None:
    neural, analytic = _complete_trials()
    trials = pd.concat([neural, analytic], ignore_index=True)
    summary = aggregate_alimu_trials(neural, analytic)
    summary.loc[0, "accel_rmse_mean"] = 123.0

    with pytest.raises(ValueError, match="canonical aggregate"):
        render_alimu_matrix_report(summary, trials, _provenance())


def test_report_rejects_artifact_path_that_escapes_provenance_run_dir() -> None:
    neural, analytic = _complete_trials()
    trials = pd.concat([neural, analytic], ignore_index=True)
    summary = aggregate_alimu_trials(neural, analytic)
    provenance = _provenance()
    provenance.pop("run_id")
    provenance["run_dir"] = "runs/allowed"
    provenance["artifacts"]["trials"] = {"path": "runs/allowed/../other/trials.csv"}
    provenance["artifacts"]["overall_summary"] = {"path": "runs/allowed/overall_summary.csv"}

    with pytest.raises(ValueError, match=r"must be under provenance\.run_dir"):
        render_alimu_matrix_report(summary, trials, provenance)


def test_report_rejects_missing_provenance_run_dir() -> None:
    neural, analytic = _complete_trials()
    trials = pd.concat([neural, analytic], ignore_index=True)
    summary = aggregate_alimu_trials(neural, analytic)
    provenance = _provenance()
    provenance.pop("run_dir")

    with pytest.raises(ValueError, match="run_dir"):
        render_alimu_matrix_report(summary, trials, provenance)


def test_report_renders_equivalent_nested_provenance_deterministically() -> None:
    neural, analytic = _complete_trials()
    trials = pd.concat([neural, analytic], ignore_index=True)
    summary = aggregate_alimu_trials(neural, analytic)
    provenance = _provenance()
    reversed_provenance = _reverse_mapping_order(provenance)

    assert isinstance(reversed_provenance, dict)
    assert render_alimu_matrix_report(summary, trials, provenance).encode() == render_alimu_matrix_report(
        summary, trials, reversed_provenance
    ).encode()


def test_report_renders_unordered_train_and_validation_sensors_deterministically() -> None:
    neural, analytic = _complete_trials()
    trials = pd.concat([neural, analytic], ignore_index=True)
    summary = aggregate_alimu_trials(neural, analytic)
    provenance = _provenance()
    reversed_sensors_provenance = _provenance()
    reversed_sensors_provenance["split_provenance"]["train_sensors"].reverse()
    reversed_sensors_provenance["split_provenance"]["validation_sensors"].reverse()

    assert render_alimu_matrix_report(summary, trials, provenance).encode() == render_alimu_matrix_report(
        summary, trials, reversed_sensors_provenance
    ).encode()


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda provenance: provenance["config_snapshots"]["matrix"].pop("sha256"), "config_snapshots"),
        (lambda provenance: provenance["inputs"]["val"].update({"sha256": "not-a-hash"}), "inputs.val"),
        (lambda provenance: provenance["artifacts"]["trials"].update({"run_id": "another-run"}), "run_id"),
    ],
)
def test_report_rejects_missing_or_mismatched_bound_provenance(mutate, match: str) -> None:
    neural, analytic = _complete_trials()
    trials = pd.concat([neural, analytic], ignore_index=True)
    summary = aggregate_alimu_trials(neural, analytic)
    provenance = _provenance()
    mutate(provenance)

    with pytest.raises(ValueError, match=match):
        render_alimu_matrix_report(summary, trials, provenance)
