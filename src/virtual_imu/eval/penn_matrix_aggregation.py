"""Hierarchical aggregation for the four-fold Penn experiment matrix."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from virtual_imu.eval.penn_loso_comparison import PENN_BASELINE_METHODS
from virtual_imu.training.penn_matrix_config import (
    FOLD_ASSIGNMENTS,
    PENN_NEURAL_METHODS,
    PENN_SEEDS,
)

CORE_METRICS = (
    "gyro_rmse",
    "gyro_mae",
    "accel_rmse",
    "accel_mae",
    "mean_axis_correlation",
    "normalized_rmse_score",
    "gyro_frequency_band_rmse",
    "accel_frequency_band_rmse",
    "gyro_coherence",
    "accel_coherence",
    "gyro_phase_delay_s",
    "accel_phase_delay_s",
    "imu_preintegration_residual",
    "delta_rotation_error_rad",
    "delta_velocity_error_m_s",
    "delta_position_proxy_error_m",
)

_SUMMARY_METRICS = CORE_METRICS[:5]
_SPECTRAL_METRICS = CORE_METRICS[6:12]
_PREINTEGRATION_METRICS = CORE_METRICS[12:]
_IDENTITY_METHOD = "identity"
_FIR_METHOD = "learned_fir_causal"
_TCN_METHOD = "tcn"
_FOLD_ORDER = tuple(FOLD_ASSIGNMENTS)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"metrics.{path} must be a mapping")
    return value


def _metric_values(source: Mapping[str, Any], names: tuple[str, ...], path: str) -> dict[str, Any]:
    missing = [name for name in names if name not in source]
    if missing:
        raise ValueError(f"metrics.{path} is missing core metrics: {missing}")
    return {name: source[name] for name in names}


def trial_row(
    fold_id: str,
    method: str,
    seed: int | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Flatten one evaluation payload into the canonical matrix trial schema."""
    payload = _mapping(metrics, "root")
    summary = _mapping(payload.get("summary"), "summary")
    spectral = _mapping(payload.get("spectral"), "spectral")
    spectral_method = _mapping(spectral.get("method"), "spectral.method")
    preintegration = _mapping(payload.get("preintegration"), "preintegration")
    preintegration_method = _mapping(
        preintegration.get("method"), "preintegration.method"
    )
    if "sample_count" not in payload:
        raise ValueError("metrics is missing sample_count")

    return {
        "fold_id": str(fold_id),
        "method": str(method),
        "seed": seed,
        "sample_count": payload["sample_count"],
        **_metric_values(summary, _SUMMARY_METRICS, "summary"),
        "normalized_rmse_score": summary.get("normalized_rmse_score", np.nan),
        **_metric_values(spectral_method, _SPECTRAL_METRICS, "spectral.method"),
        **_metric_values(
            preintegration_method,
            _PREINTEGRATION_METRICS,
            "preintegration.method",
        ),
    }


def _require_columns(table: pd.DataFrame, table_name: str) -> None:
    required = ["fold_id", "method", "seed", "sample_count", *CORE_METRICS]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{table_name} is missing columns: {missing}")


def _expected_axis(
    methods: tuple[str, ...], seeds: tuple[int, ...] | None
) -> set[tuple[str, str, int | None]]:
    if seeds is None:
        return {
            (fold_id, method, None)
            for fold_id in FOLD_ASSIGNMENTS
            for method in methods
        }
    return {
        (fold_id, method, seed)
        for fold_id in FOLD_ASSIGNMENTS
        for method in methods
        for seed in seeds
    }


def _validate_neural_trials(table: pd.DataFrame) -> None:
    _require_columns(table, "neural_trials")
    validated_seeds: list[int] = []
    for seed in table["seed"]:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ValueError("neural_trials seed values must be configured integers")
        validated_seed = int(seed)
        if validated_seed not in PENN_SEEDS:
            raise ValueError("neural_trials seed values must be configured integers")
        validated_seeds.append(validated_seed)

    duplicate = table.duplicated(["fold_id", "method", "seed"], keep=False)
    if duplicate.any():
        keys = table.loc[duplicate, ["fold_id", "method", "seed"]].to_dict("records")
        raise ValueError(f"neural_trials contains duplicate trial axes: {keys}")

    actual = {
        (str(row.fold_id), str(row.method), validated_seeds[index])
        for index, row in enumerate(
            table[["fold_id", "method"]].itertuples(index=False)
        )
    }
    expected = _expected_axis(PENN_NEURAL_METHODS, PENN_SEEDS)
    if len(table) != 60 or actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "neural_trials must contain the complete 60-row fold/method/seed axes; "
            f"missing={missing}, extra={extra}"
        )


def _validate_analytic_folds(table: pd.DataFrame) -> None:
    _require_columns(table, "analytic_folds")
    if table["seed"].notna().any():
        raise ValueError("analytic_folds seed values must be null")
    duplicate = table.duplicated(["fold_id", "method"], keep=False)
    if duplicate.any():
        keys = table.loc[duplicate, ["fold_id", "method"]].to_dict("records")
        raise ValueError(f"analytic_folds contains duplicate fold/method axes: {keys}")

    actual = {
        (str(row.fold_id), str(row.method), None)
        for row in table[["fold_id", "method"]].itertuples(index=False)
    }
    expected = _expected_axis(PENN_BASELINE_METHODS, None)
    if len(table) != 16 or actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "analytic_folds must contain the complete 16-row fold/method axes; "
            f"missing={missing}, extra={extra}"
        )


def _validate_numeric_values(neural: pd.DataFrame, analytic: pd.DataFrame) -> None:
    columns = ["sample_count", *(metric for metric in CORE_METRICS if metric != "normalized_rmse_score")]
    for name, table in (("neural_trials", neural), ("analytic_folds", analytic)):
        try:
            values = table[columns].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} core values must be numeric") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite core values")
        if np.any(values[:, 0] <= 0):
            raise ValueError(f"{name} sample_count values must be positive")


def _score_against_fold_identity(
    neural: pd.DataFrame, analytic: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = (
        analytic.loc[
            analytic["method"].eq(_IDENTITY_METHOD),
            ["fold_id", "gyro_rmse", "accel_rmse"],
        ]
        .set_index("fold_id")
        .rename(
            columns={
                "gyro_rmse": "identity_gyro_rmse",
                "accel_rmse": "identity_accel_rmse",
            }
        )
    )
    if (identity <= 0).any().any():
        raise ValueError("Identity fold RMSE values must be positive")

    scored = []
    for table in (neural, analytic):
        result = table.drop(columns=["normalized_rmse_score"]).join(identity, on="fold_id")
        result["normalized_rmse_score"] = 0.5 * (
            result["gyro_rmse"] / result["identity_gyro_rmse"]
            + result["accel_rmse"] / result["identity_accel_rmse"]
        )
        result = result.drop(columns=["identity_gyro_rmse", "identity_accel_rmse"])
        scored.append(result)
    return scored[0], scored[1]


def _fold_rows(neural: pd.DataFrame, analytic: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold_id, method), group in neural.groupby(["fold_id", "method"], sort=False):
        row: dict[str, Any] = {
            "fold_id": fold_id,
            "method": method,
            "sample_count": int(group["sample_count"].iloc[0]),
            "seed_count": len(group),
        }
        for metric in CORE_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_seed_std"] = float(group[metric].std(ddof=1))
        rows.append(row)

    for record in analytic.to_dict(orient="records"):
        row = {
            "fold_id": record["fold_id"],
            "method": record["method"],
            "sample_count": int(record["sample_count"]),
            "seed_count": 0,
        }
        for metric in CORE_METRICS:
            row[f"{metric}_mean"] = float(record[metric])
            row[f"{metric}_seed_std"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _improvement(candidate: pd.Series, reference: pd.Series) -> pd.Series:
    return 100.0 * (reference - candidate) / reference


def _overall_rows(fold_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, group in fold_summary.groupby("method", sort=False):
        worst_fold = min(
            _FOLD_ORDER,
            key=lambda fold_id: (
                -float(
                    group.loc[
                        group["fold_id"].eq(fold_id),
                        "normalized_rmse_score_mean",
                    ].iloc[0]
                ),
                _FOLD_ORDER.index(fold_id),
            ),
        )
        row: dict[str, Any] = {
            "method": method,
            "fold_count": len(group),
            "trial_count": int(group["seed_count"].sum()) or len(group),
            "sample_count": int(group["sample_count"].sum()),
            "worst_fold": worst_fold,
        }
        for metric in CORE_METRICS:
            fold_means = group[f"{metric}_mean"]
            seed_stds = group[f"{metric}_seed_std"]
            row[f"{metric}_mean"] = float(fold_means.mean())
            row[f"{metric}_fold_std"] = float(fold_means.std(ddof=1))
            row[f"{metric}_seed_std_mean"] = (
                float(seed_stds.mean()) if seed_stds.notna().any() else np.nan
            )
        rows.append(row)

    overall = pd.DataFrame(rows)
    score = overall.set_index("method")["normalized_rmse_score_mean"]
    references = {
        "identity": _IDENTITY_METHOD,
        "fir": _FIR_METHOD,
        "tcn": _TCN_METHOD,
    }
    for label, method in references.items():
        column = f"normalized_rmse_improvement_vs_{label}_pct"
        overall[column] = (
            _improvement(overall["normalized_rmse_score_mean"], score[method])
            if method in score.index
            else np.nan
        )

    overall = overall.sort_values(
        ["normalized_rmse_score_mean", "method"], kind="stable"
    ).reset_index(drop=True)
    overall.insert(0, "rank", np.arange(1, len(overall) + 1))
    return overall


def aggregate_matrix_trials(
    neural_trials: pd.DataFrame,
    analytic_folds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate seeds within folds, then macro-average the four fold means."""
    neural = neural_trials.copy()
    analytic = analytic_folds.copy()
    _validate_neural_trials(neural)
    _validate_analytic_folds(analytic)
    _validate_numeric_values(neural, analytic)
    neural, analytic = _score_against_fold_identity(neural, analytic)

    fold_summary = _fold_rows(neural, analytic)
    overall_summary = _overall_rows(fold_summary)
    return fold_summary, overall_summary


def _markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_table_artifacts(
    table: pd.DataFrame,
    base_path: str | Path,
    *,
    title: str,
) -> dict[str, Path]:
    """Write one aggregation table to CSV, JSON, and Markdown."""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": base.with_suffix(".csv"),
        "json": base.with_suffix(".json"),
        "markdown": base.with_suffix(".md"),
    }
    table.to_csv(paths["csv"], index=False)
    paths["json"].write_text(table.to_json(orient="records", indent=2), encoding="utf-8")

    columns = list(table.columns)
    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_markdown_value(value) for value in row) + " |")
    paths["markdown"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    for path in paths.values():
        logger.info("Wrote Penn matrix aggregation artifact: {}", path)
    return paths


def write_trials(trials: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Write the canonical trial table in all supported formats."""
    return write_table_artifacts(
        trials, Path(output_dir) / "trials", title="Penn Matrix Trials"
    )


def write_fold_summary(fold_summary: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Write the fold-level hierarchical summary in all supported formats."""
    return write_table_artifacts(
        fold_summary,
        Path(output_dir) / "fold_summary",
        title="Penn Matrix Fold Summary",
    )


def write_overall_summary(
    overall_summary: pd.DataFrame, output_dir: str | Path
) -> dict[str, Path]:
    """Write the ranked macro-fold summary in all supported formats."""
    return write_table_artifacts(
        overall_summary,
        Path(output_dir) / "overall_summary",
        title="Penn Matrix Overall Summary",
    )


def write_matrix_aggregation(
    trials: pd.DataFrame,
    fold_summary: pd.DataFrame,
    overall_summary: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, dict[str, Path]]:
    """Write trials, fold summary, and overall summary artifacts."""
    return {
        "trials": write_trials(trials, output_dir),
        "fold_summary": write_fold_summary(fold_summary, output_dir),
        "overall_summary": write_overall_summary(overall_summary, output_dir),
    }
