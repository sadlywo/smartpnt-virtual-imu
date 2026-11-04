"""Single-holdout aggregation for the ALiMU leave-sensor experiment matrix."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from virtual_imu.eval.penn_matrix_aggregation import CORE_METRICS
from virtual_imu.training.alimu_matrix_config import (
    ALIMU_HOLDOUT_SENSOR,
    ALIMU_NEURAL_METHODS,
    ALIMU_SEEDS,
)

ALIMU_ANALYTIC_METHODS = (
    "identity",
    "mean_bias",
    "channel_affine",
    "learned_fir_causal",
)

_SUMMARY_METRICS = CORE_METRICS[:5]
_SPECTRAL_METRICS = CORE_METRICS[6:12]
_PREINTEGRATION_METRICS = CORE_METRICS[12:]
_IDENTITY_METHOD = "identity"
_FIR_METHOD = "learned_fir_causal"
_TCN_METHOD = "tcn"
_METHOD_ORDER = (*ALIMU_NEURAL_METHODS, *ALIMU_ANALYTIC_METHODS)
_METHOD_POSITION = {method: index for index, method in enumerate(_METHOD_ORDER)}


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"metrics.{path} must be a mapping")
    return value


def _metric_values(source: Mapping[str, Any], names: tuple[str, ...], path: str) -> dict[str, Any]:
    missing = [name for name in names if name not in source]
    if missing:
        raise ValueError(f"metrics.{path} is missing core metrics: {missing}")
    return {name: source[name] for name in names}


def alimu_trial_row(
    holdout_sensor: str,
    method: str,
    seed: int | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Flatten one canonical evaluator payload into an ALiMU trial row."""
    payload = _mapping(metrics, "root")
    summary = _mapping(payload.get("summary"), "summary")
    spectral = _mapping(payload.get("spectral"), "spectral")
    spectral_method = _mapping(spectral.get("method"), "spectral.method")
    preintegration = _mapping(payload.get("preintegration"), "preintegration")
    preintegration_method = _mapping(preintegration.get("method"), "preintegration.method")
    if "sample_count" not in payload:
        raise ValueError("metrics is missing sample_count")

    return {
        "holdout_sensor": str(holdout_sensor),
        "method": str(method),
        "seed": seed,
        "sample_count": payload["sample_count"],
        **_metric_values(summary, _SUMMARY_METRICS, "summary"),
        "normalized_rmse_score": summary.get("normalized_rmse_score", np.nan),
        **_metric_values(spectral_method, _SPECTRAL_METRICS, "spectral.method"),
        **_metric_values(preintegration_method, _PREINTEGRATION_METRICS, "preintegration.method"),
    }


def _require_columns(table: pd.DataFrame, table_name: str) -> None:
    required = ["holdout_sensor", "method", "seed", "sample_count", *CORE_METRICS]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"{table_name} is missing columns: {missing}")


def _validate_neural_trials(table: pd.DataFrame) -> None:
    _require_columns(table, "neural_trials")
    seeds: list[int] = []
    for seed in table["seed"]:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or int(seed) not in ALIMU_SEEDS:
            raise ValueError("neural_trials seed values must be configured integers")
        seeds.append(int(seed))
    duplicate = table.duplicated(["holdout_sensor", "method", "seed"], keep=False)
    if duplicate.any():
        keys = table.loc[duplicate, ["holdout_sensor", "method", "seed"]].to_dict("records")
        raise ValueError(f"neural_trials contains duplicate trial axes: {keys}")
    expected = {
        (ALIMU_HOLDOUT_SENSOR, method, seed)
        for method in ALIMU_NEURAL_METHODS
        for seed in ALIMU_SEEDS
    }
    actual = {
        (str(row.holdout_sensor), str(row.method), seeds[index])
        for index, row in enumerate(table[["holdout_sensor", "method"]].itertuples(index=False))
    }
    if len(table) != len(expected) or actual != expected:
        raise ValueError(
            "neural_trials must contain the complete 15-row holdout_sensor/method/seed axes; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _validate_analytic_trials(table: pd.DataFrame) -> None:
    _require_columns(table, "analytic_trials")
    if table["seed"].notna().any():
        raise ValueError("analytic_trials seed values must be null")
    duplicate = table.duplicated(["holdout_sensor", "method"], keep=False)
    if duplicate.any():
        keys = table.loc[duplicate, ["holdout_sensor", "method"]].to_dict("records")
        raise ValueError(f"analytic_trials contains duplicate trial axes: {keys}")
    expected = {(ALIMU_HOLDOUT_SENSOR, method) for method in ALIMU_ANALYTIC_METHODS}
    actual = set(
        table[["holdout_sensor", "method"]].astype(str).itertuples(index=False, name=None)
    )
    if len(table) != len(expected) or actual != expected:
        raise ValueError(
            "analytic_trials must contain the complete 4-row holdout_sensor/method axes; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _validate_numeric_values(neural: pd.DataFrame, analytic: pd.DataFrame) -> None:
    columns = ["sample_count", *(metric for metric in CORE_METRICS if metric != "normalized_rmse_score")]
    for name, table in (("neural_trials", neural), ("analytic_trials", analytic)):
        try:
            values = table[columns].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} core values must be numeric") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite core values")
        if np.any(values[:, 0] <= 0):
            raise ValueError(f"{name} sample_count values must be positive")
    sample_counts = pd.concat([neural["sample_count"], analytic["sample_count"]])
    if sample_counts.nunique() != 1:
        raise ValueError("all ALiMU trials must share one sample_count")


def _score_against_identity(
    neural: pd.DataFrame, analytic: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = analytic.loc[analytic["method"].eq(_IDENTITY_METHOD)].iloc[0]
    gyro_reference = float(identity["gyro_rmse"])
    accel_reference = float(identity["accel_rmse"])
    if gyro_reference <= 0 or accel_reference <= 0:
        raise ValueError("Identity RMSE values must be positive")
    scored: list[pd.DataFrame] = []
    for table in (neural, analytic):
        result = table.copy()
        result["normalized_rmse_score"] = 0.5 * (
            result["gyro_rmse"] / gyro_reference
            + result["accel_rmse"] / accel_reference
        )
        scored.append(result)
    return scored[0], scored[1]


def _improvement(candidate: pd.Series, reference: pd.Series) -> pd.Series:
    return 100.0 * (reference - candidate) / reference


def _summary_rows(neural: pd.DataFrame, analytic: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in ALIMU_NEURAL_METHODS:
        group = neural.loc[neural["method"].eq(method)].sort_values("seed", kind="stable")
        row: dict[str, Any] = {
            "holdout_sensor": ALIMU_HOLDOUT_SENSOR,
            "method": method,
            "sample_count": int(group["sample_count"].iloc[0]),
            "trial_count": len(group),
            "seed_count": len(group),
        }
        for metric in CORE_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_seed_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    for method in ALIMU_ANALYTIC_METHODS:
        record = analytic.loc[analytic["method"].eq(method)].iloc[0]
        row = {
            "holdout_sensor": ALIMU_HOLDOUT_SENSOR,
            "method": method,
            "sample_count": int(record["sample_count"]),
            "trial_count": 1,
            "seed_count": 0,
        }
        for metric in CORE_METRICS:
            row[f"{metric}_mean"] = float(record[metric])
            row[f"{metric}_seed_std"] = np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    score = summary.set_index("method")["normalized_rmse_score_mean"]
    for label, method in (("identity", _IDENTITY_METHOD), ("fir", _FIR_METHOD), ("tcn", _TCN_METHOD)):
        summary[f"normalized_rmse_improvement_vs_{label}_pct"] = _improvement(
            summary["normalized_rmse_score_mean"], score[method]
        )
    summary = summary.assign(_method_order=summary["method"].map(_METHOD_POSITION))
    summary = summary.sort_values(
        ["normalized_rmse_score_mean", "_method_order"], kind="stable"
    ).drop(columns="_method_order").reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def aggregate_alimu_trials(
    neural_trials: pd.DataFrame, analytic_trials: pd.DataFrame
) -> pd.DataFrame:
    """Validate and summarize the fixed ALiMU single-holdout experiment matrix."""
    neural = neural_trials.copy()
    analytic = analytic_trials.copy()
    _validate_neural_trials(neural)
    _validate_analytic_trials(analytic)
    _validate_numeric_values(neural, analytic)
    neural, analytic = _score_against_identity(neural, analytic)
    return _summary_rows(neural, analytic)


def _reject_fold_fields(table: pd.DataFrame, table_name: str) -> None:
    forbidden = [column for column in table.columns if "fold" in str(column).lower()]
    if forbidden:
        raise ValueError(f"{table_name} must not contain fold fields: {forbidden}")


def _canonical_summary_from_trials(trials: pd.DataFrame) -> pd.DataFrame:
    _reject_fold_fields(trials, "trials")
    _require_columns(trials, "trials")
    expected_methods = set(_METHOD_ORDER)
    actual_methods = set(trials["method"].astype(str))
    if actual_methods != expected_methods or len(trials) != 19:
        raise ValueError("trials must contain exactly the canonical 15 neural and 4 analytic rows")
    neural = trials.loc[trials["method"].isin(ALIMU_NEURAL_METHODS)]
    analytic = trials.loc[trials["method"].isin(ALIMU_ANALYTIC_METHODS)]
    return aggregate_alimu_trials(neural, analytic)


def validate_alimu_matrix_artifacts(
    trials: pd.DataFrame, overall_summary: pd.DataFrame
) -> pd.DataFrame:
    """Return the canonical aggregate after proving supplied artifacts match it exactly."""
    _reject_fold_fields(overall_summary, "overall_summary")
    canonical = _canonical_summary_from_trials(trials)
    if list(overall_summary.columns) != list(canonical.columns):
        raise ValueError("overall_summary must equal the canonical aggregate derived from trials")
    supplied = overall_summary.sort_values("rank", kind="stable").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(supplied, canonical, check_exact=True)
    except AssertionError as exc:
        raise ValueError(
            "overall_summary must equal the canonical aggregate derived from trials"
        ) from exc
    return canonical


def _ordered_trials(trials: pd.DataFrame) -> pd.DataFrame:
    ordered = trials.copy()
    ordered["_method_order"] = ordered["method"].map(_METHOD_POSITION)
    ordered["_seed_order"] = pd.to_numeric(ordered["seed"], errors="coerce").fillna(-1)
    return ordered.sort_values(
        ["holdout_sensor", "_method_order", "_seed_order"], kind="stable"
    ).drop(columns=["_method_order", "_seed_order"]).reset_index(drop=True)


def _markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_alimu_table_artifacts(
    table: pd.DataFrame, base_path: str | Path, *, title: str
) -> dict[str, Path]:
    """Write one ALiMU table to CSV, JSON, and Markdown and log every write."""
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
        logger.info("WRITES kind=alimu_matrix_aggregation path={}", path)
    return paths


def write_alimu_trials(trials: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Write canonical ALiMU trial artifacts in deterministic order."""
    return write_alimu_table_artifacts(
        _ordered_trials(trials), Path(output_dir) / "trials", title="ALiMU Matrix Trials"
    )


def write_alimu_overall_summary(
    overall_summary: pd.DataFrame, output_dir: str | Path
) -> dict[str, Path]:
    """Write the ranked ALiMU single-holdout summary in all supported formats."""
    ordered = overall_summary.sort_values("rank", kind="stable").reset_index(drop=True)
    return write_alimu_table_artifacts(
        ordered, Path(output_dir) / "overall_summary", title="ALiMU Matrix Overall Summary"
    )


def write_alimu_matrix_aggregation(
    trials: pd.DataFrame, overall_summary: pd.DataFrame, output_dir: str | Path
) -> dict[str, dict[str, Path]]:
    """Write only ALiMU trial and single-holdout summary artifacts."""
    canonical = validate_alimu_matrix_artifacts(trials, overall_summary)
    return {
        "trials": write_alimu_trials(trials, output_dir),
        "overall_summary": write_alimu_overall_summary(canonical, output_dir),
    }
