from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from virtual_imu.eval.penn_loso_comparison import PENN_BASELINE_METHODS
from virtual_imu.eval.penn_matrix_aggregation import (
    CORE_METRICS,
    aggregate_matrix_trials,
    trial_row,
    write_matrix_aggregation,
)
from virtual_imu.eval.penn_matrix_report import (
    render_penn_matrix_report,
    write_penn_matrix_report,
)
from virtual_imu.training.penn_matrix_config import (
    FOLD_ASSIGNMENTS,
    PENN_NEURAL_METHODS,
    PENN_SEEDS,
)

FOLDS = tuple(FOLD_ASSIGNMENTS)


def _metrics(value: float, sample_count: int = 100) -> dict[str, Any]:
    return {
        "sample_count": sample_count,
        "summary": {
            "gyro_rmse": value,
            "gyro_mae": value + 0.1,
            "accel_rmse": 2.0 * value,
            "accel_mae": 2.0 * value + 0.1,
            "mean_axis_correlation": 1.0 - value / 100.0,
        },
        "spectral": {
            "method": {
                "gyro_frequency_band_rmse": value + 0.2,
                "accel_frequency_band_rmse": value + 0.3,
                "gyro_coherence": 0.9 - value / 100.0,
                "accel_coherence": 0.8 - value / 100.0,
                "gyro_phase_delay_s": value / 1000.0,
                "accel_phase_delay_s": value / 900.0,
            }
        },
        "preintegration": {
            "method": {
                "imu_preintegration_residual": value + 0.4,
                "delta_rotation_error_rad": value + 0.5,
                "delta_velocity_error_m_s": value + 0.6,
                "delta_position_proxy_error_m": value + 0.7,
            }
        },
    }


def _complete_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    identity_values = (10.0, 20.0, 30.0, 40.0)
    identity_by_fold = {fold_id: identity_values[index] for index, fold_id in enumerate(FOLDS)}
    bias_drift_values = {
        FOLDS[0]: (1.0, 2.0, 3.0),
        FOLDS[1]: (2.0, 4.0, 6.0),
        FOLDS[2]: (9.0, 12.0, 15.0),
        FOLDS[3]: (4.0, 8.0, 12.0),
    }
    neural_rows = []
    sample_count_values = (10, 100, 1_000, 10_000)
    sample_counts = {
        fold_id: sample_count_values[index] for index, fold_id in enumerate(FOLDS)
    }
    for fold_id in FOLDS:
        for method_index, method in enumerate(PENN_NEURAL_METHODS):
            values = (
                bias_drift_values[fold_id]
                if method == "bias_drift"
                else tuple(
                    identity_by_fold[fold_id] * (0.45 + method_index * 0.05)
                    for _ in PENN_SEEDS
                )
            )
            for index, seed in enumerate(PENN_SEEDS):
                value = values[index]
                neural_rows.append(
                    trial_row(
                        fold_id,
                        method,
                        seed,
                        _metrics(value, sample_counts[fold_id]),
                    )
                )

    baseline_ratios = {
        "identity": 1.0,
        "mean_bias": 0.8,
        "channel_affine": 0.7,
        "learned_fir_causal": 0.6,
    }
    analytic_rows = [
        trial_row(
            fold_id,
            method,
            None,
            _metrics(identity_by_fold[fold_id] * baseline_ratios[method], sample_counts[fold_id]),
        )
        for fold_id in FOLDS
        for method in PENN_BASELINE_METHODS
    ]
    return pd.DataFrame(neural_rows), pd.DataFrame(analytic_rows)


def _report_provenance() -> dict[str, Any]:
    return {
        "experiment_id": "penn-matrix-test",
        "git_commit": "abc1234",
        "config_snapshot": "configs/penn_loso_matrix_3seed.yaml",
        "artifacts": {"run_dir": "runs/penn-matrix-test"},
        "split_provenance": {
            fold_id: {
                split: list(sequences)
                for split, sequences in assignment.items()
            }
            for fold_id, assignment in FOLD_ASSIGNMENTS.items()
        },
    }


def _report_value(value: float) -> str:
    return f"{value:.6g}"


def test_trial_row_extracts_canonical_nested_metrics() -> None:
    row = trial_row("test_af", "bias_drift", PENN_SEEDS[0], _metrics(2.0, 37))

    assert row["fold_id"] == "test_af"
    assert row["method"] == "bias_drift"
    assert row["seed"] == PENN_SEEDS[0]
    assert row["sample_count"] == 37
    assert row["gyro_rmse"] == 2.0
    assert row["accel_coherence"] == pytest.approx(0.78)
    assert row["delta_position_proxy_error_m"] == pytest.approx(2.7)
    assert set(CORE_METRICS) <= set(row)
    assert np.isnan(row["normalized_rmse_score"])


def test_hierarchical_aggregation_uses_seed_then_fold_macro_statistics() -> None:
    neural, analytic = _complete_tables()

    fold_summary, overall = aggregate_matrix_trials(neural, analytic)

    fold_means = np.array([2.0, 4.0, 12.0, 8.0])
    seed_stds = np.array([1.0, 2.0, 3.0, 4.0])
    row = overall.loc[overall["method"].eq("bias_drift")].iloc[0]
    assert row["gyro_rmse_mean"] == pytest.approx(np.mean(fold_means))
    assert row["gyro_rmse_fold_std"] == pytest.approx(np.std(fold_means, ddof=1))
    assert row["gyro_rmse_seed_std_mean"] == pytest.approx(np.mean(seed_stds))
    assert row["worst_fold"] == FOLDS[2]

    fold_row = fold_summary.loc[
        fold_summary["method"].eq("bias_drift")
        & fold_summary["fold_id"].eq(FOLDS[2])
    ].iloc[0]
    assert fold_row["gyro_rmse_mean"] == 12.0
    assert fold_row["gyro_rmse_seed_std"] == 3.0
    assert fold_row["sample_count"] == 1_000

    pooled_by_samples = np.average(fold_means, weights=[10, 100, 1_000, 10_000])
    assert row["gyro_rmse_mean"] != pytest.approx(pooled_by_samples)
    assert row["gyro_rmse_fold_std"] != pytest.approx(
        np.std([1, 2, 3, 2, 4, 6, 9, 12, 15, 4, 8, 12], ddof=1)
    )


def test_analytic_baselines_use_four_folds_without_fake_seed_variance() -> None:
    neural, analytic = _complete_tables()

    fold_summary, overall = aggregate_matrix_trials(neural, analytic)

    identity_folds = fold_summary.loc[fold_summary["method"].eq("identity")]
    identity = overall.loc[overall["method"].eq("identity")].iloc[0]
    assert len(identity_folds) == 4
    assert identity_folds["seed_count"].eq(0).all()
    assert identity_folds["gyro_rmse_seed_std"].isna().all()
    assert np.isnan(identity["gyro_rmse_seed_std_mean"])
    assert identity["gyro_rmse_fold_std"] == pytest.approx(np.std([10, 20, 30, 40], ddof=1))


def test_normalized_score_ranking_and_reference_improvements() -> None:
    neural, analytic = _complete_tables()

    fold_summary, overall = aggregate_matrix_trials(neural, analytic)

    identity_folds = fold_summary.loc[fold_summary["method"].eq("identity")]
    assert identity_folds["normalized_rmse_score_mean"].tolist() == pytest.approx([1.0] * 4)
    assert overall["normalized_rmse_score_mean"].is_monotonic_increasing
    assert overall["rank"].tolist() == list(range(1, len(overall) + 1))

    identity = overall.loc[overall["method"].eq("identity")].iloc[0]
    fir = overall.loc[overall["method"].eq("learned_fir_causal")].iloc[0]
    tcn = overall.loc[overall["method"].eq("tcn")].iloc[0]
    assert identity["normalized_rmse_improvement_vs_identity_pct"] == 0.0
    assert fir["normalized_rmse_improvement_vs_identity_pct"] == pytest.approx(40.0)
    assert tcn["normalized_rmse_improvement_vs_tcn_pct"] == 0.0
    assert fir["normalized_rmse_improvement_vs_fir_pct"] == 0.0


def test_worst_fold_tie_uses_canonical_fold_order_for_reversed_input() -> None:
    neural, analytic = _complete_tables()

    _, overall = aggregate_matrix_trials(
        neural.iloc[::-1].reset_index(drop=True),
        analytic.iloc[::-1].reset_index(drop=True),
    )

    tcn = overall.loc[overall["method"].eq("tcn")].iloc[0]
    assert tcn["worst_fold"] == FOLDS[0]


@pytest.mark.parametrize(
    "invalid_seed",
    [float(PENN_SEEDS[0]), PENN_SEEDS[0] + 0.5, str(PENN_SEEDS[0]), True],
)
def test_rejects_non_integer_seed_values_without_coercion(invalid_seed: object) -> None:
    neural, analytic = _complete_tables()
    neural["seed"] = neural["seed"].astype(object)
    neural.at[0, "seed"] = invalid_seed

    with pytest.raises(ValueError, match=r"seed.*configured integers"):
        aggregate_matrix_trials(neural, analytic)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "nonfinite"])
def test_rejects_incomplete_duplicate_or_nonfinite_inputs(problem: str) -> None:
    neural, analytic = _complete_tables()
    if problem == "missing":
        neural = neural.iloc[:-1].copy()
        match = "complete.*60"
    elif problem == "duplicate":
        neural = pd.concat([neural, neural.iloc[[0]]], ignore_index=True)
        match = "duplicate"
    else:
        analytic.loc[0, "gyro_rmse"] = np.inf
        match = "non-finite"

    with pytest.raises(ValueError, match=match):
        aggregate_matrix_trials(neural, analytic)


def test_rejects_non_null_analytic_seed() -> None:
    neural, analytic = _complete_tables()
    analytic.loc[0, "seed"] = PENN_SEEDS[0]

    with pytest.raises(ValueError, match=r"analytic.*seed.*null"):
        aggregate_matrix_trials(neural, analytic)


def test_writes_csv_json_and_markdown_for_all_tables(tmp_path: Path) -> None:
    neural, analytic = _complete_tables()
    fold_summary, overall = aggregate_matrix_trials(neural, analytic)
    trials = pd.concat([neural, analytic], ignore_index=True)

    paths = write_matrix_aggregation(trials, fold_summary, overall, tmp_path)

    assert set(paths) == {"trials", "fold_summary", "overall_summary"}
    for name, formats in paths.items():
        assert set(formats) == {"csv", "json", "markdown"}
        assert formats["csv"] == tmp_path / f"{name}.csv"
        assert all(path.exists() for path in formats.values())
        assert json.loads(formats["json"].read_text(encoding="utf-8"))
        assert formats["markdown"].read_text(encoding="utf-8").startswith("# ")


def test_renders_complete_chinese_matrix_report_and_writes_it(tmp_path: Path) -> None:
    neural, analytic = _complete_tables()
    folds, overall = aggregate_matrix_trials(neural, analytic)
    trials = pd.concat([neural, analytic], ignore_index=True)
    provenance = _report_provenance()

    report = render_penn_matrix_report(overall, folds, trials, provenance)

    for phrase in (
        "四折 LOSO",
        "FIR-Residual",
        "最差 fold",
        "预积分",
        "均值 ± fold 标准差",
        "不能等同于跨传感器泛化",
        "tcn",
        "bias_only",
        "drift_only",
        "bias_drift",
        "learned_fir_causal",
        "abc1234",
        "各 fold 排名变化",
        "seed 稳定性",
        "陀螺相干性",
        "陀螺相位延迟 (s)",
        "旋转增量误差 (rad)",
        "产物归属以 provenance 为准",
        "as, bf",
        "bs",
        "af",
    ):
        assert phrase in report

    bias_drift_folds = folds.loc[folds["method"].eq("bias_drift")]
    seed_stability = bias_drift_folds["normalized_rmse_score_seed_std"].mean()
    assert _report_value(seed_stability) in report
    assert _report_value(overall["gyro_coherence_mean"].max()) in report
    assert _report_value(overall["gyro_phase_delay_s_mean"].abs().min()) in report
    assert _report_value(overall["delta_rotation_error_rad_mean"].min()) in report
    assert f"| {FOLDS[0]} | bias_drift |" in report
    spectral_section = report.split("## 频谱、相干性与相位发现", maxsplit=1)[1].split(
        "## 预积分权衡", maxsplit=1
    )[0]
    assert "最差方法" in spectral_section
    assert "最差值" in spectral_section

    report_without_phase = render_penn_matrix_report(
        overall.drop(columns=["accel_phase_delay_s_mean"]),
        folds,
        trials,
        provenance,
    )
    phase_line = next(
        line
        for line in report_without_phase.splitlines()
        if line.startswith("| 加速度计相位延迟 (s) |")
    )
    assert phase_line == "| 加速度计相位延迟 (s) | 不可用 | 不可用 | 不可用 | 不可用 |"

    path = write_penn_matrix_report(
        overall,
        folds,
        trials,
        provenance,
        tmp_path / "report.md",
    )
    assert path == tmp_path / "report.md"
    assert path.read_text(encoding="utf-8") == report


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", r"split_provenance.*required"),
        ("missing_fold", r"split_provenance.*fold IDs"),
        ("overlap", r"split_provenance.*overlap"),
        ("inconsistent_scope", r"split_provenance.*sequence scope"),
    ],
)
def test_rejects_missing_or_inconsistent_core_split_provenance(
    mutation: str, match: str
) -> None:
    neural, analytic = _complete_tables()
    folds, overall = aggregate_matrix_trials(neural, analytic)
    trials = pd.concat([neural, analytic], ignore_index=True)
    provenance = _report_provenance()
    if mutation == "missing":
        provenance.pop("split_provenance")
    elif mutation == "missing_fold":
        provenance["split_provenance"].pop(FOLDS[-1])
    elif mutation == "overlap":
        provenance["split_provenance"][FOLDS[0]]["validation"] = ["as"]
    else:
        provenance["split_provenance"][FOLDS[0]]["train"] = ["extra"]

    with pytest.raises(ValueError, match=match):
        render_penn_matrix_report(overall, folds, trials, provenance)


@pytest.mark.parametrize("fold_count", [3, 5])
def test_report_requires_exactly_the_canonical_four_fold_ids(fold_count: int) -> None:
    neural, analytic = _complete_tables()
    folds, overall = aggregate_matrix_trials(neural, analytic)
    trials = pd.concat([neural, analytic], ignore_index=True)
    provenance = _report_provenance()
    if fold_count == 3:
        removed = FOLDS[-1]
        folds = folds.loc[folds["fold_id"].ne(removed)].copy()
        trials = trials.loc[trials["fold_id"].ne(removed)].copy()
        provenance["split_provenance"].pop(removed)
    else:
        extra_fold = "test_extra"
        extra_folds = folds.loc[folds["fold_id"].eq(FOLDS[0])].copy()
        extra_folds["fold_id"] = extra_fold
        folds = pd.concat([folds, extra_folds], ignore_index=True)
        extra_trials = trials.loc[trials["fold_id"].eq(FOLDS[0])].copy()
        extra_trials["fold_id"] = extra_fold
        trials = pd.concat([trials, extra_trials], ignore_index=True)
        provenance["split_provenance"][extra_fold] = {
            "train": ["as", "bf"],
            "validation": ["bs"],
            "test": ["af"],
        }

    with pytest.raises(ValueError, match=r"exactly the canonical four fold IDs"):
        render_penn_matrix_report(overall, folds, trials, provenance)


def test_report_is_deterministic_for_reversed_input_order() -> None:
    neural, analytic = _complete_tables()
    folds, overall = aggregate_matrix_trials(neural, analytic)
    trials = pd.concat([neural, analytic], ignore_index=True)
    provenance = _report_provenance()
    expected = render_penn_matrix_report(overall, folds, trials, provenance)
    reversed_provenance = {
        key: provenance[key] for key in reversed(tuple(provenance))
    }
    reversed_provenance["split_provenance"] = {
        fold_id: {
            split: assignment[split] for split in reversed(tuple(assignment))
        }
        for fold_id, assignment in reversed(
            tuple(provenance["split_provenance"].items())
        )
    }

    actual = render_penn_matrix_report(
        overall.iloc[::-1].reset_index(drop=True),
        folds.iloc[::-1].reset_index(drop=True),
        trials.iloc[::-1].reset_index(drop=True),
        reversed_provenance,
    )

    assert actual == expected


def test_rejects_noncanonical_split_assignment() -> None:
    neural, analytic = _complete_tables()
    folds, overall = aggregate_matrix_trials(neural, analytic)
    trials = pd.concat([neural, analytic], ignore_index=True)
    provenance = _report_provenance()
    provenance["split_provenance"][FOLDS[0]] = {
        "train": ["as", "bs"],
        "validation": ["bf"],
        "test": ["af"],
    }

    with pytest.raises(ValueError, match=r"split_provenance.*FOLD_ASSIGNMENTS"):
        render_penn_matrix_report(overall, folds, trials, provenance)


@pytest.mark.parametrize(
    ("table_name", "column", "value"),
    [
        ("overall", "rank", "first"),
        ("overall", "gyro_rmse_mean", np.inf),
        ("folds", "normalized_rmse_score_mean", "bad"),
        ("folds", "accel_rmse_mean", np.inf),
        ("trials", "sample_count", np.inf),
        ("trials", "seed", "seed"),
    ],
)
def test_report_rejects_malformed_or_nonfinite_required_values(
    table_name: str, column: str, value: object
) -> None:
    neural, analytic = _complete_tables()
    folds, overall = aggregate_matrix_trials(neural, analytic)
    trials = pd.concat([neural, analytic], ignore_index=True)
    tables = {"overall": overall, "folds": folds, "trials": trials}
    table = tables[table_name].copy()
    if isinstance(value, str):
        table[column] = table[column].astype(object)
    elif isinstance(value, float) and not np.isfinite(value):
        table[column] = table[column].astype(float)
    table.loc[table.index[0], column] = value
    tables[table_name] = table

    with pytest.raises(ValueError, match=rf"{table_name}.*numeric.*finite"):
        render_penn_matrix_report(
            tables["overall"], tables["folds"], tables["trials"], _report_provenance()
        )


@pytest.mark.parametrize("table_name", ["folds", "trials"])
def test_report_rejects_duplicate_rows_used_by_report(table_name: str) -> None:
    neural, analytic = _complete_tables()
    folds, overall = aggregate_matrix_trials(neural, analytic)
    trials = pd.concat([neural, analytic], ignore_index=True)
    if table_name == "folds":
        folds = pd.concat([folds, folds.iloc[[0]]], ignore_index=True)
    else:
        trials = pd.concat([trials, trials.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match=rf"{table_name}.*duplicate"):
        render_penn_matrix_report(overall, folds, trials, _report_provenance())


def test_report_rejects_reduced_one_seed_trial_matrix() -> None:
    neural, analytic = _complete_tables()
    folds, overall = aggregate_matrix_trials(neural, analytic)
    trials = pd.concat([neural, analytic], ignore_index=True)
    reduced = trials.loc[
        trials["seed"].isna() | trials["seed"].eq(PENN_SEEDS[0])
    ].copy()
    assert len(reduced) == 36

    with pytest.raises(ValueError, match=r"trials.*exact 60-row neural matrix"):
        render_penn_matrix_report(overall, folds, reduced, _report_provenance())


def test_report_accepts_exact_neural_trials_without_optional_analytic_rows() -> None:
    neural, analytic = _complete_tables()
    folds, overall = aggregate_matrix_trials(neural, analytic)

    report = render_penn_matrix_report(overall, folds, neural, _report_provenance())

    assert "seed 稳定性" in report
    assert "| identity | 0 | 不可用 |" in report


@pytest.mark.parametrize(
    ("method", "seed_count"),
    [("tcn", 1), ("identity", 1)],
)
def test_report_reconciles_fold_seed_counts_with_trial_matrix(
    method: str, seed_count: int
) -> None:
    neural, analytic = _complete_tables()
    folds, overall = aggregate_matrix_trials(neural, analytic)
    trials = pd.concat([neural, analytic], ignore_index=True)
    folds.loc[folds["method"].eq(method), "seed_count"] = seed_count

    with pytest.raises(ValueError, match=r"folds.*seed_count.*trials"):
        render_penn_matrix_report(overall, folds, trials, _report_provenance())


def test_report_rejects_noncanonical_worst_fold() -> None:
    neural, analytic = _complete_tables()
    folds, overall = aggregate_matrix_trials(neural, analytic)
    trials = pd.concat([neural, analytic], ignore_index=True)
    overall.loc[overall["method"].eq("tcn"), "worst_fold"] = "not_a_fold"

    with pytest.raises(ValueError, match=r"overall\.worst_fold.*canonical"):
        render_penn_matrix_report(overall, folds, trials, _report_provenance())


@pytest.mark.parametrize("missing_value", [np.nan, np.inf])
def test_optional_nonfinite_phase_metric_renders_unavailable(missing_value: float) -> None:
    neural, analytic = _complete_tables()
    folds, overall = aggregate_matrix_trials(neural, analytic)
    trials = pd.concat([neural, analytic], ignore_index=True)
    overall["accel_phase_delay_s_mean"] = missing_value

    report = render_penn_matrix_report(overall, folds, trials, _report_provenance())
    phase_line = next(
        line
        for line in report.splitlines()
        if line.startswith("| 加速度计相位延迟 (s) |")
    )
    assert phase_line == "| 加速度计相位延迟 (s) | 不可用 | 不可用 | 不可用 | 不可用 |"


def test_report_values_are_derived_from_all_input_frames() -> None:
    neural, analytic = _complete_tables()
    folds, overall = aggregate_matrix_trials(neural, analytic)
    trials = pd.concat([neural, analytic], ignore_index=True)
    provenance = _report_provenance()

    original = render_penn_matrix_report(overall, folds, trials, provenance)
    changed_overall = overall.copy()
    changed_overall.loc[
        changed_overall["method"].eq("bias_drift"), "gyro_rmse_mean"
    ] = 123.456
    changed = render_penn_matrix_report(
        changed_overall,
        folds,
        trials,
        provenance,
    )

    assert original != changed
    assert "123.456" in changed
    assert "123.456" not in original

    changed_folds = folds.copy()
    changed_folds.loc[
        changed_folds["method"].eq("bias_drift"),
        "normalized_rmse_score_seed_std",
    ] = 98.7654
    changed_folds.loc[
        changed_folds["fold_id"].eq(FOLDS[0])
        & changed_folds["method"].eq("identity"),
        "normalized_rmse_score_mean",
    ] = -10.0
    changed = render_penn_matrix_report(overall, changed_folds, trials, provenance)
    identity_rank = int(overall.loc[overall["method"].eq("identity"), "rank"].iloc[0])
    assert f"| {FOLDS[0]} | identity | 1 | {identity_rank} | {1 - identity_rank} |" in changed
    assert "98.7654" in changed
    assert "98.7654" not in original

    changed_supplementary = overall.copy()
    hybrid = changed_supplementary["method"].eq("fir_bias_drift_residual")
    changed_supplementary.loc[hybrid, "gyro_frequency_band_rmse_mean"] = 87.6543
    changed_supplementary.loc[hybrid, "gyro_coherence_mean"] = -9.87654
    changed_supplementary.loc[hybrid, "gyro_phase_delay_s_mean"] = 7.65432
    changed_supplementary.loc[hybrid, "delta_rotation_error_rad_mean"] = 76.5432
    changed = render_penn_matrix_report(
        changed_supplementary, folds, trials, provenance
    )
    for value in ("87.6543", "-9.87654", "7.65432", "76.5432"):
        assert value in changed
        assert value not in original
