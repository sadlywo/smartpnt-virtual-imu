"""Chinese Markdown reporting for the Penn four-fold experiment matrix."""

from __future__ import annotations

import json
from collections.abc import Mapping
from numbers import Real
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

_CANONICAL_FOLD_ORDER = tuple(FOLD_ASSIGNMENTS)
_CANONICAL_FOLD_IDS = frozenset(_CANONICAL_FOLD_ORDER)
_CANONICAL_METHOD_ORDER = (*PENN_NEURAL_METHODS, *PENN_BASELINE_METHODS)
_CANONICAL_METHOD_IDS = frozenset(_CANONICAL_METHOD_ORDER)
_METHOD_ORDER = {
    method: index for index, method in enumerate(_CANONICAL_METHOD_ORDER)
}
_ABLATION_METHODS = ("tcn", "bias_only", "drift_only", "bias_drift")
_HYBRID_METHOD = "fir_bias_drift_residual"
_FIR_METHOD = "learned_fir_causal"
_DISPLAY_NAMES = {
    _HYBRID_METHOD: "FIR-Residual (fir_bias_drift_residual)",
    _FIR_METHOD: "frozen FIR (learned_fir_causal)",
}

_OVERALL_CORE = (
    "rank",
    "method",
    "normalized_rmse_score_mean",
    "normalized_rmse_score_fold_std",
    "gyro_rmse_mean",
    "gyro_rmse_fold_std",
    "accel_rmse_mean",
    "accel_rmse_fold_std",
    "worst_fold",
)
_FOLD_CORE = (
    "fold_id",
    "method",
    "sample_count",
    "seed_count",
    "normalized_rmse_score_mean",
    "gyro_rmse_mean",
    "accel_rmse_mean",
)
_TRIAL_CORE = ("fold_id", "method", "seed", "sample_count")
_OVERALL_NUMERIC = tuple(
    column for column in _OVERALL_CORE if column not in {"method", "worst_fold"}
)
_FOLD_NUMERIC = tuple(
    column for column in _FOLD_CORE if column not in {"fold_id", "method"}
)

_FINDING_METRICS = (
    ("gyro_frequency_band_rmse_mean", "陀螺频带 RMSE", "min"),
    ("accel_frequency_band_rmse_mean", "加速度计频带 RMSE", "min"),
    ("gyro_coherence_mean", "陀螺相干性", "max"),
    ("accel_coherence_mean", "加速度计相干性", "max"),
    ("gyro_phase_delay_s_mean", "陀螺相位延迟 (s)", "abs_min"),
    ("accel_phase_delay_s_mean", "加速度计相位延迟 (s)", "abs_min"),
)
_PREINTEGRATION_METRICS = (
    ("imu_preintegration_residual_mean", "IMU 预积分残差"),
    ("delta_rotation_error_rad_mean", "旋转增量误差 (rad)"),
    ("delta_velocity_error_m_s_mean", "速度增量误差 (m/s)"),
    ("delta_position_proxy_error_m_mean", "位置代理误差 (m)"),
)


def _require_columns(table: pd.DataFrame, name: str, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{name} is missing core columns: {missing}")


def _validate_canonical_fold_ids(folds: pd.DataFrame, trials: pd.DataFrame) -> None:
    fold_ids = set(folds["fold_id"].astype(str))
    trial_fold_ids = set(trials["fold_id"].astype(str))
    if fold_ids != _CANONICAL_FOLD_IDS or trial_fold_ids != _CANONICAL_FOLD_IDS:
        raise ValueError(
            "folds and trials must contain exactly the canonical four fold IDs: "
            f"{sorted(_CANONICAL_FOLD_IDS)}"
        )


def _is_finite_real(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value)


def _require_finite_numeric(
    table: pd.DataFrame,
    name: str,
    columns: tuple[str, ...],
    *,
    nullable: tuple[str, ...] = (),
) -> None:
    for column in columns:
        invalid = [
            value
            for value in table[column]
            if not (column in nullable and pd.isna(value)) and not _is_finite_real(value)
        ]
        if invalid:
            raise ValueError(
                f"{name} required numeric columns must contain finite numeric values; "
                f"invalid column={column}"
            )


def _validate_report_frames(
    overall: pd.DataFrame, folds: pd.DataFrame, trials: pd.DataFrame
) -> None:
    _require_finite_numeric(overall, "overall", _OVERALL_NUMERIC)
    _require_finite_numeric(folds, "folds", _FOLD_NUMERIC)
    _require_finite_numeric(
        trials, "trials", ("seed", "sample_count"), nullable=("seed",)
    )

    if overall["method"].duplicated().any() or overall["rank"].duplicated().any():
        raise ValueError("overall contains duplicate method or rank rows")
    if folds.duplicated(["fold_id", "method"]).any():
        raise ValueError("folds contains duplicate fold/method rows")
    if trials.duplicated(["fold_id", "method", "seed"]).any():
        raise ValueError("trials contains duplicate fold/method/seed rows")

    ranks = overall["rank"].tolist()
    if any(not _is_finite_real(rank) or not float(rank).is_integer() for rank in ranks):
        raise ValueError("overall required numeric rank values must be finite integers")
    if sorted(int(rank) for rank in ranks) != list(range(1, len(overall) + 1)):
        raise ValueError("overall rank must contain contiguous unique integers")

    overall_methods = set(overall["method"].astype(str))
    fold_methods = set(folds["method"].astype(str))
    trial_methods = set(trials["method"].astype(str))
    if overall_methods != _CANONICAL_METHOD_IDS or fold_methods != _CANONICAL_METHOD_IDS:
        raise ValueError("report frames must contain exactly the canonical method IDs")
    allowed_trial_method_sets = {
        frozenset(PENN_NEURAL_METHODS),
        _CANONICAL_METHOD_IDS,
    }
    if frozenset(trial_methods) not in allowed_trial_method_sets:
        raise ValueError(
            "trials must contain all neural methods and either zero or all analytic methods"
        )
    expected_fold_methods = {
        (fold_id, method)
        for fold_id in _CANONICAL_FOLD_ORDER
        for method in _CANONICAL_METHOD_ORDER
    }
    actual_fold_methods = set(
        folds[["fold_id", "method"]].astype(str).itertuples(index=False, name=None)
    )
    if actual_fold_methods != expected_fold_methods:
        raise ValueError("folds must contain every canonical fold/method row exactly once")
    trial_fold_methods = set(
        trials[["fold_id", "method"]].astype(str).itertuples(index=False, name=None)
    )
    expected_trial_fold_methods = {
        (fold_id, method)
        for fold_id in _CANONICAL_FOLD_ORDER
        for method in trial_methods
    }
    if trial_fold_methods != expected_trial_fold_methods:
        raise ValueError("trials must contain every included canonical fold/method axis")

    if (overall["rank"] <= 0).any() or (folds["sample_count"] <= 0).any():
        raise ValueError("overall/folds required numeric count values must be positive")
    if (folds["seed_count"] < 0).any() or (trials["sample_count"] <= 0).any():
        raise ValueError("folds/trials required numeric count values must be non-negative")
    if any(
        not _is_finite_real(value) or not float(value).is_integer()
        for value in folds["seed_count"]
    ):
        raise ValueError("folds required numeric seed_count values must be finite integers")
    trial_seeds = trials["seed"].dropna()
    if any(
        not _is_finite_real(value) or not float(value).is_integer()
        for value in trial_seeds
    ):
        raise ValueError("trials required numeric seed values must be finite integers")
    if trials.loc[trials["method"].isin(PENN_NEURAL_METHODS), "seed"].isna().any():
        raise ValueError("trials neural method seed values must be present")
    if trials.loc[trials["method"].isin(PENN_BASELINE_METHODS), "seed"].notna().any():
        raise ValueError("trials analytic method seed values must be null")

    neural_trials = trials.loc[trials["method"].isin(PENN_NEURAL_METHODS)]
    expected_neural_axes = {
        (fold_id, method, seed)
        for fold_id in _CANONICAL_FOLD_ORDER
        for method in PENN_NEURAL_METHODS
        for seed in PENN_SEEDS
    }
    actual_neural_axes = {
        (str(row.fold_id), str(row.method), int(row.seed))
        for row in neural_trials[["fold_id", "method", "seed"]].itertuples(
            index=False
        )
    }
    if len(neural_trials) != len(expected_neural_axes) or actual_neural_axes != expected_neural_axes:
        raise ValueError(
            "trials must contain the exact 60-row neural matrix with every "
            "canonical fold/method/PENN_SEEDS axis once"
        )

    analytic_trials = trials.loc[trials["method"].isin(PENN_BASELINE_METHODS)]
    if not analytic_trials.empty:
        expected_analytic_axes = {
            (fold_id, method)
            for fold_id in _CANONICAL_FOLD_ORDER
            for method in PENN_BASELINE_METHODS
        }
        actual_analytic_axes = set(
            analytic_trials[["fold_id", "method"]]
            .astype(str)
            .itertuples(index=False, name=None)
        )
        if (
            len(analytic_trials) != len(expected_analytic_axes)
            or actual_analytic_axes != expected_analytic_axes
        ):
            raise ValueError(
                "trials optional analytic rows must contain the exact 16-row "
                "canonical fold/method matrix"
            )

    expected_seed_counts = {
        method: len(PENN_SEEDS) if method in PENN_NEURAL_METHODS else 0
        for method in _CANONICAL_METHOD_ORDER
    }
    if any(
        int(row.seed_count) != expected_seed_counts[str(row.method)]
        for row in folds[["method", "seed_count"]].itertuples(index=False)
    ):
        raise ValueError("folds seed_count values must reconcile with the trials matrix")

    worst_fold_pairs = set(
        overall[["worst_fold", "method"]].astype(str).itertuples(index=False, name=None)
    )
    if any(fold_id not in _CANONICAL_FOLD_IDS for fold_id, _ in worst_fold_pairs):
        raise ValueError("overall.worst_fold values must be canonical fold IDs")
    if not worst_fold_pairs <= actual_fold_methods:
        raise ValueError("overall.worst_fold must reference an existing fold/method row")
    if any(group["sample_count"].nunique() != 1 for _, group in folds.groupby("fold_id")):
        raise ValueError("folds sample_count must be consistent within each fold")


def _format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return "不可用"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "不可用"
        return f"{float(value):.6g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value).replace("|", "\\|").replace("\n", " ")


def _mean_std(row: pd.Series, metric: str) -> str:
    return (
        f"{_format_value(row.get(f'{metric}_mean'))} ± "
        f"{_format_value(row.get(f'{metric}_fold_std'))}"
    )


def _method_name(method: Any) -> str:
    value = str(method)
    return _DISPLAY_NAMES.get(value, value)


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(_format_value(value) for value in row) + " |" for row in rows
    )
    return lines


def _row_for_method(overall: pd.DataFrame, method: str) -> pd.Series | None:
    selected = overall.loc[overall["method"].eq(method)]
    return None if selected.empty else selected.iloc[0]


def _validate_split_provenance(
    provenance: Mapping[str, Any], folds: pd.DataFrame, trials: pd.DataFrame
) -> dict[str, dict[str, tuple[str, ...]]]:
    raw = provenance.get("split_provenance")
    if not isinstance(raw, Mapping):
        raise ValueError("provenance.split_provenance is required and must be a mapping")

    fold_ids = _CANONICAL_FOLD_ORDER
    expected_folds = set(fold_ids)
    trial_folds = set(trials["fold_id"].astype(str))
    if trial_folds != expected_folds:
        raise ValueError("folds and trials must contain the same fold IDs")
    if set(str(key) for key in raw) != expected_folds:
        raise ValueError("provenance.split_provenance fold IDs must match input frames")

    validated: dict[str, dict[str, tuple[str, ...]]] = {}
    expected_splits = {"train", "validation", "test"}
    sequence_scope: set[str] | None = None
    test_sequences: list[str] = []
    for fold_id in fold_ids:
        assignment = raw.get(fold_id)
        if not isinstance(assignment, Mapping) or set(assignment) != expected_splits:
            raise ValueError(
                "provenance.split_provenance assignments must contain exactly "
                "train, validation, and test"
            )
        validated_assignment: dict[str, tuple[str, ...]] = {}
        for split in ("train", "validation", "test"):
            values = assignment[split]
            if (
                not isinstance(values, (list, tuple))
                or not values
                or any(not isinstance(value, str) or not value.strip() for value in values)
                or len(set(values)) != len(values)
            ):
                raise ValueError(
                    "provenance.split_provenance split values must be non-empty "
                    "lists of unique sequence IDs"
                )
            validated_assignment[split] = tuple(values)

        split_sets = [set(validated_assignment[split]) for split in expected_splits]
        if any(split_sets[left] & split_sets[right] for left in range(3) for right in range(left + 1, 3)):
            raise ValueError("provenance.split_provenance splits overlap within a fold")
        current_scope = set().union(*split_sets)
        if sequence_scope is None:
            sequence_scope = current_scope
        elif current_scope != sequence_scope:
            raise ValueError(
                "provenance.split_provenance sequence scope must be consistent across folds"
            )
        test_sequences.extend(validated_assignment["test"])
        validated[fold_id] = validated_assignment

    if sequence_scope is None or set(test_sequences) != sequence_scope or len(test_sequences) != len(
        sequence_scope
    ):
        raise ValueError(
            "provenance.split_provenance must hold out every sequence exactly once"
        )
    for fold_id, assignment in validated.items():
        expected_assignment = FOLD_ASSIGNMENTS[fold_id]
        if any(
            assignment[split] != tuple(expected_assignment[split])
            for split in ("train", "validation", "test")
        ):
            raise ValueError(
                "provenance.split_provenance must exactly match FOLD_ASSIGNMENTS "
                f"for {fold_id}"
            )
    return validated


def _scope_section(
    folds: pd.DataFrame, split_provenance: Mapping[str, Mapping[str, tuple[str, ...]]]
) -> list[str]:
    rows: list[tuple[Any, ...]] = []
    for fold_id in _CANONICAL_FOLD_ORDER:
        assignment = split_provenance[str(fold_id)]
        fold_rows = folds.loc[folds["fold_id"].eq(fold_id)]
        sample_count = fold_rows["sample_count"].iloc[0]
        rows.append(
            (
                fold_id,
                ", ".join(assignment["train"]),
                ", ".join(assignment["validation"]),
                ", ".join(assignment["test"]),
                sample_count,
            )
        )
    return [
        "## 四折 LOSO 范围与划分",
        "",
        f"本报告覆盖 {folds['fold_id'].nunique()} 个 fold, 样本数来自 fold 汇总表。",
        "",
        *_table(("fold", "训练序列", "验证序列", "测试序列", "样本数"), rows),
    ]


def _overall_section(overall: pd.DataFrame) -> list[str]:
    ordered = overall.assign(
        _method_order=overall["method"].map(_METHOD_ORDER)
    ).sort_values(["rank", "_method_order"], kind="stable")
    rows = [
        (
            row["rank"],
            _method_name(row["method"]),
            _mean_std(row, "normalized_rmse_score"),
            _mean_std(row, "gyro_rmse"),
            _mean_std(row, "accel_rmse"),
        )
        for _, row in ordered.iterrows()
    ]
    return [
        "## 总体排名",
        "",
        "表中均为四个 fold 的宏平均; 表达形式为均值 ± fold 标准差。",
        "",
        *_table(
            ("rank", "方法", "归一化 RMSE", "陀螺 RMSE", "加速度计 RMSE"),
            rows,
        ),
    ]


def _ablation_section(overall: pd.DataFrame) -> list[str]:
    rows = []
    for method in _ABLATION_METHODS:
        row = _row_for_method(overall, method)
        rows.append(
            (
                method,
                "不可用" if row is None else row["rank"],
                "不可用" if row is None else _mean_std(row, "normalized_rmse_score"),
                "不可用" if row is None else _mean_std(row, "gyro_rmse"),
                "不可用" if row is None else _mean_std(row, "accel_rmse"),
            )
        )
    return [
        "## 四路消融",
        "",
        *_table(("方法", "总体 rank", "归一化 RMSE", "陀螺 RMSE", "加速度计 RMSE"), rows),
    ]


def _comparison_section(overall: pd.DataFrame) -> list[str]:
    candidate = _row_for_method(overall, _HYBRID_METHOD)
    rows: list[tuple[Any, ...]] = []
    for reference_method in (_FIR_METHOD, "bias_drift"):
        reference = _row_for_method(overall, reference_method)
        if candidate is None or reference is None:
            rows.append((_method_name(reference_method), "不可用", "不可用", "不可用"))
            continue
        rows.append(
            (
                _method_name(reference_method),
                candidate["normalized_rmse_score_mean"]
                - reference["normalized_rmse_score_mean"],
                candidate["gyro_rmse_mean"] - reference["gyro_rmse_mean"],
                candidate["accel_rmse_mean"] - reference["accel_rmse_mean"],
            )
        )
    return [
        "## FIR-Residual 混合方法对比",
        "",
        "FIR-Residual 与 frozen FIR 及 bias_drift 的差值为前者减参考方法; 负值表示误差更低。",
        "",
        *_table(("参考方法", "归一化 RMSE 差", "陀螺 RMSE 差", "加速度计 RMSE 差"), rows),
    ]


def _fold_rank_section(overall: pd.DataFrame, folds: pd.DataFrame) -> list[str]:
    overall_ranks = overall.set_index("method")["rank"].to_dict()
    rows: list[tuple[Any, ...]] = []
    for fold_id in _CANONICAL_FOLD_ORDER:
        group = folds.loc[folds["fold_id"].eq(fold_id)]
        ranked = group.assign(
            _method_order=group["method"].map(_METHOD_ORDER)
        ).sort_values(
            ["normalized_rmse_score_mean", "_method_order"], kind="stable"
        ).reset_index(drop=True)
        for index, row in ranked.iterrows():
            overall_rank = overall_ranks.get(row["method"])
            fold_rank = index + 1
            change = "不可用" if overall_rank is None else fold_rank - overall_rank
            rows.append((fold_id, _method_name(row["method"]), fold_rank, overall_rank, change))
    return [
        "## 各 fold 排名变化",
        "",
        "变化量 = fold rank - 总体 rank。",
        "",
        *_table(("fold", "方法", "fold rank", "总体 rank", "变化量"), rows),
    ]


def _stability_section(folds: pd.DataFrame, trials: pd.DataFrame) -> list[str]:
    rows: list[tuple[Any, ...]] = []
    for method in _CANONICAL_METHOD_ORDER:
        method_folds = folds.loc[folds["method"].eq(method)]
        seeds = trials.loc[trials["method"].eq(method), "seed"].dropna()
        column = "normalized_rmse_score_seed_std"
        stability = (
            method_folds[column].mean()
            if column in method_folds and method_folds[column].notna().any()
            else np.nan
        )
        rows.append((_method_name(method), seeds.nunique(), stability))
    return [
        "## seed 稳定性",
        "",
        *_table(("方法", "seed 数", "fold 内归一化 RMSE seed 标准差均值"), rows),
    ]


def _worst_fold_section(overall: pd.DataFrame, folds: pd.DataFrame) -> list[str]:
    rows = []
    for _, row in overall.sort_values(["rank", "method"], kind="stable").iterrows():
        match = folds.loc[
            folds["method"].eq(row["method"])
            & folds["fold_id"].eq(row["worst_fold"])
        ]
        score = np.nan if match.empty else match["normalized_rmse_score_mean"].iloc[0]
        rows.append((_method_name(row["method"]), row["worst_fold"], score))
    return [
        "## 最差 fold",
        "",
        *_table(("方法", "最差 fold", "该 fold 归一化 RMSE"), rows),
    ]


def _best_row(overall: pd.DataFrame, column: str, direction: str) -> pd.Series | None:
    if column not in overall:
        return None
    values = pd.to_numeric(overall[column], errors="coerce")
    finite = np.isfinite(values)
    if not finite.any():
        return None
    candidates = overall.loc[finite].copy()
    candidates["_selection"] = values.loc[finite]
    if direction == "abs_min":
        candidates["_selection"] = candidates["_selection"].abs()
    candidates["_method_order"] = candidates["method"].map(_METHOD_ORDER)
    ascending = direction != "max"
    return candidates.sort_values(
        ["_selection", "_method_order"], ascending=[ascending, True], kind="stable"
    ).iloc[0]


def _worst_row(overall: pd.DataFrame, column: str, direction: str) -> pd.Series | None:
    if column not in overall:
        return None
    reverse_direction = "min" if direction == "max" else "max"
    if direction == "abs_min":
        values = pd.to_numeric(overall[column], errors="coerce")
        finite = np.isfinite(values)
        if not finite.any():
            return None
        candidates = overall.loc[finite].copy()
        candidates["_selection"] = values.loc[finite].abs()
        candidates["_method_order"] = candidates["method"].map(_METHOD_ORDER)
        return candidates.sort_values(
            ["_selection", "_method_order"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
    return _best_row(overall, column, reverse_direction)


def _findings_section(overall: pd.DataFrame) -> list[str]:
    rows = []
    for column, label, direction in _FINDING_METRICS:
        best = _best_row(overall, column, direction)
        worst = _worst_row(overall, column, direction)
        rows.append(
            (
                label,
                "不可用" if best is None else _method_name(best["method"]),
                "不可用" if best is None else best[column],
                "不可用" if worst is None else _method_name(worst["method"]),
                "不可用" if worst is None else worst[column],
            )
        )
    return [
        "## 频谱、相干性与相位发现",
        "",
        "频带 RMSE 取最小, 相干性取最大, 相位延迟取绝对值最小。",
        "",
        *_table(("指标", "最佳方法", "最佳值", "最差方法", "最差值"), rows),
    ]


def _preintegration_section(overall: pd.DataFrame) -> list[str]:
    rows = []
    for column, label in _PREINTEGRATION_METRICS:
        best = _best_row(overall, column, "min")
        worst = _worst_row(overall, column, "min")
        rows.append(
            (
                label,
                "不可用" if best is None else _method_name(best["method"]),
                "不可用" if best is None else best[column],
                "不可用" if worst is None else _method_name(worst["method"]),
                "不可用" if worst is None else worst[column],
            )
        )
    return [
        "## 预积分权衡",
        "",
        "下表给出可用列上的精确最佳/最差对比; 需与短时信号 RMSE 排名联合解读。",
        "",
        *_table(("预积分指标", "最佳方法", "最佳值", "最差方法", "最差值"), rows),
    ]


def _provenance_section(provenance: Mapping[str, Any]) -> list[str]:
    rows = []
    for key in sorted(provenance, key=str):
        value = provenance[key]
        rendered = (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (Mapping, list, tuple))
            else value
        )
        rows.append((key, rendered))
    if not rows:
        rows.append(("来源记录", "不可用"))
    return [
        "## 产物与溯源",
        "",
        "本报告仅解读传入的 trials、fold 汇总和 overall 汇总; 产物归属以 provenance 为准。",
        "",
        *_table(("溯源字段", "值"), rows),
    ]


def render_penn_matrix_report(
    overall: pd.DataFrame,
    folds: pd.DataFrame,
    trials: pd.DataFrame,
    provenance: Mapping[str, Any],
) -> str:
    """Render a data-driven Chinese Markdown report for the Penn matrix."""
    _require_columns(overall, "overall", _OVERALL_CORE)
    _require_columns(folds, "folds", _FOLD_CORE)
    _require_columns(trials, "trials", _TRIAL_CORE)
    _validate_canonical_fold_ids(folds, trials)
    _validate_report_frames(overall, folds, trials)
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    split_provenance = _validate_split_provenance(provenance, folds, trials)

    sections = [
        ["# Penn 虚拟 IMU 实验矩阵报告"],
        _scope_section(folds, split_provenance),
        _overall_section(overall),
        _ablation_section(overall),
        _comparison_section(overall),
        _fold_rank_section(overall, folds),
        _stability_section(folds, trials),
        _worst_fold_section(overall, folds),
        _findings_section(overall),
        _preintegration_section(overall),
        [
            "## 局限性",
            "",
            "该结果是固定 Penn 序列边界下的 LOSO 评估, 不能等同于跨传感器泛化。",
            "fold 数、seed 数和样本分布均受传入表限制; “不可用”表示补充指标未随表提供。",
        ],
        _provenance_section(provenance),
    ]
    return "\n\n".join("\n".join(section) for section in sections) + "\n"


def write_penn_matrix_report(
    overall: pd.DataFrame,
    folds: pd.DataFrame,
    trials: pd.DataFrame,
    provenance: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Render and write a Penn matrix Markdown report."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_penn_matrix_report(overall, folds, trials, provenance),
        encoding="utf-8",
    )
    logger.info("Wrote Penn matrix report: {}", output_path)
    return output_path
