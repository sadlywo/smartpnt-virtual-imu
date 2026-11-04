"""Chinese Markdown report for the ALiMU strict leave-sensor matrix."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from virtual_imu.eval.alimu_matrix_aggregation import (
    ALIMU_ANALYTIC_METHODS,
    validate_alimu_matrix_artifacts,
)
from virtual_imu.training.alimu_matrix_config import (
    ALIMU_HOLDOUT_SENSOR,
    ALIMU_NEURAL_METHODS,
)

_METHOD_ORDER = (*ALIMU_NEURAL_METHODS, *ALIMU_ANALYTIC_METHODS)
_METHOD_POSITION = {method: index for index, method in enumerate(_METHOD_ORDER)}
_HYBRID_METHOD = "fir_bias_drift_residual"
_FIR_METHOD = "learned_fir_causal"
_ABLATION_METHODS = ("tcn", "bias_only", "drift_only", "bias_drift")
_REQUIRED_SUMMARY = (
    "rank",
    "holdout_sensor",
    "method",
    "sample_count",
    "trial_count",
    "seed_count",
    "normalized_rmse_score_mean",
    "normalized_rmse_score_seed_std",
    "gyro_rmse_mean",
    "gyro_rmse_seed_std",
    "accel_rmse_mean",
    "accel_rmse_seed_std",
)
_SPECTRAL_METRICS = (
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


def _is_finite_real(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value)


def _format(value: Any) -> str:
    if value is None or pd.isna(value):
        return "不可用"
    if isinstance(value, Real) and not _is_finite_real(value):
        return "不可用"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.6g}" if isinstance(value, (float, np.floating)) else str(value)


def _table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(_format(value) for value in row) + " |" for row in rows),
    ]


def _mean_std(row: pd.Series, metric: str) -> str:
    return f"{_format(row.get(metric + '_mean'))} ± {_format(row.get(metric + '_seed_std'))}"


def _validate_provenance(provenance: Mapping[str, Any]) -> None:
    run_id = provenance.get("run_id")
    run_dir = provenance.get("run_dir")
    if not _nonempty_string(run_dir):
        raise ValueError("provenance.run_dir must be a non-empty path")
    resolved_run_dir = Path(run_dir).resolve()
    _validate_hashed_references(provenance.get("config_snapshots"), "config_snapshots", ("matrix", "base"))
    _validate_hashed_references(provenance, "split_manifest", ("split_manifest",))
    _validate_hashed_references(provenance.get("inputs"), "inputs", ("train", "val", "test"))
    _validate_artifact_references(provenance.get("artifacts"), run_id, resolved_run_dir)
    split = provenance.get("split_provenance")
    if not isinstance(split, Mapping):
        raise ValueError("provenance.split_provenance is required and must be a mapping")
    if split.get("mode") != "leave_sensor_out":
        raise ValueError("provenance.split_provenance.mode must be leave_sensor_out")
    if split.get("holdout_sensor") != ALIMU_HOLDOUT_SENSOR:
        raise ValueError("provenance.split_provenance.holdout_sensor must be MPU9250")
    expected_train = {"BMX160", "LSM9DS1"}
    for key in ("train_sensors", "validation_sensors"):
        value = split.get(key)
        if not isinstance(value, (list, tuple)) or set(value) != expected_train:
            raise ValueError(f"provenance.split_provenance.{key} must be BMX160 and LSM9DS1")
    if set(split.get("test_sensors", ())) != {ALIMU_HOLDOUT_SENSOR}:
        raise ValueError("provenance.split_provenance.test_sensors must be MPU9250")
    if split.get("target_sensor") != "VN100" or split.get("sample_rate_hz") != 40:
        raise ValueError("provenance.split_provenance target must be VN100 at 40 Hz")


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_hashed_reference(value: Any, name: str) -> None:
    if not isinstance(value, Mapping) or not _nonempty_string(value.get("path")):
        raise ValueError(f"provenance.{name} must contain a non-empty path")
    digest = value.get("sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        raise ValueError(f"provenance.{name} must contain a 64-character sha256")


def _validate_hashed_references(
    parent: Any, name: str, keys: tuple[str, ...]
) -> None:
    if name == "split_manifest":
        _validate_hashed_reference(parent.get("split_manifest") if isinstance(parent, Mapping) else None, name)
        return
    if not isinstance(parent, Mapping):
        raise ValueError(f"provenance.{name} must be a mapping")
    for key in keys:
        _validate_hashed_reference(parent.get(key), f"{name}.{key}")


def _validate_artifact_references(artifacts: Any, run_id: Any, run_dir: Path) -> None:
    if not isinstance(artifacts, Mapping):
        raise ValueError("provenance.artifacts must be a mapping")
    for key in ("trials", "overall_summary"):
        artifact = artifacts.get(key)
        if not isinstance(artifact, Mapping) or not _nonempty_string(artifact.get("path")):
            raise ValueError(f"provenance.artifacts.{key} must contain a non-empty path")
        if _nonempty_string(run_id):
            if artifact.get("run_id") != run_id:
                raise ValueError(f"provenance.artifacts.{key}.run_id must match provenance.run_id")
        if not Path(artifact["path"]).resolve().is_relative_to(run_dir):
            raise ValueError(f"provenance.artifacts.{key}.path must be under provenance.run_dir")


def _canonicalize_provenance_value(value: Any, path: tuple[str, ...] = ()) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _canonicalize_provenance_value(item, (*path, str(key)))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        canonical = [_canonicalize_provenance_value(item, path) for item in value]
        if path in {
            ("split_provenance", "train_sensors"),
            ("split_provenance", "validation_sensors"),
        }:
            return sorted(canonical, key=str)
        return canonical
    return value


def _format_provenance_value(value: Any, path: tuple[str, ...] = ()) -> str:
    value = _canonicalize_provenance_value(value, path)
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, default=str, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return str(value)


def _validate_summary(summary: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_SUMMARY if column not in summary]
    if missing:
        raise ValueError(f"overall_summary is missing core columns: {missing}")
    fold_columns = [column for column in summary if "fold" in column.lower()]
    if fold_columns:
        raise ValueError(f"overall_summary must not contain fold columns: {fold_columns}")
    if len(summary) != len(_METHOD_ORDER) or summary["method"].duplicated().any():
        raise ValueError("overall_summary must contain each canonical method exactly once")
    if set(summary["method"].astype(str)) != set(_METHOD_ORDER):
        raise ValueError("overall_summary must contain the exact ALiMU method axes")
    if set(summary["holdout_sensor"].astype(str)) != {ALIMU_HOLDOUT_SENSOR}:
        raise ValueError("overall_summary holdout_sensor must be MPU9250")
    ranks = summary["rank"].tolist()
    if any(not _is_finite_real(value) or not float(value).is_integer() for value in ranks):
        raise ValueError("overall_summary rank values must be finite integers")
    if sorted(int(value) for value in ranks) != list(range(1, len(summary) + 1)):
        raise ValueError("overall_summary rank must be contiguous")
    for column in ("sample_count", "trial_count", "seed_count", "normalized_rmse_score_mean", "gyro_rmse_mean", "accel_rmse_mean"):
        if any(not _is_finite_real(value) for value in summary[column]):
            raise ValueError(f"overall_summary {column} values must be finite")
    if summary["sample_count"].nunique() != 1:
        raise ValueError("overall_summary sample_count must be shared by every method")


def _validate_inputs(
    summary: pd.DataFrame, trials: pd.DataFrame, provenance: Mapping[str, Any]
) -> pd.DataFrame:
    _validate_summary(summary)
    _validate_provenance(provenance)
    return validate_alimu_matrix_artifacts(trials, summary)


def _row(summary: pd.DataFrame, method: str) -> pd.Series:
    return summary.loc[summary["method"].eq(method)].iloc[0]


def _best_worst(summary: pd.DataFrame, column: str, direction: str) -> tuple[pd.Series | None, pd.Series | None]:
    if column not in summary:
        return None, None
    values = pd.to_numeric(summary[column], errors="coerce")
    candidates = summary.loc[np.isfinite(values)].copy()
    if candidates.empty:
        return None, None
    candidates["_value"] = values.loc[candidates.index]
    if direction == "abs_min":
        candidates["_selection"] = candidates["_value"].abs()
    else:
        candidates["_selection"] = candidates["_value"]
    candidates["_method"] = candidates["method"].map(_METHOD_POSITION)
    ascending = direction != "max"
    best = candidates.sort_values(["_selection", "_method"], ascending=[ascending, True], kind="stable").iloc[0]
    worst = candidates.sort_values(["_selection", "_method"], ascending=[not ascending, True], kind="stable").iloc[0]
    return best, worst


def render_alimu_matrix_report(
    overall_summary: pd.DataFrame, trials: pd.DataFrame, provenance: Mapping[str, Any]
) -> str:
    """Render a deterministic Chinese report from ALiMU summary, trials, and provenance."""
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    summary = _validate_inputs(overall_summary.copy(), trials, provenance)
    ordered = summary.sort_values("rank", kind="stable")
    ranking_rows = [
        (
            row["rank"],
            row["method"],
            _mean_std(row, "normalized_rmse_score"),
            _mean_std(row, "gyro_rmse"),
            _mean_std(row, "accel_rmse"),
        )
        for _, row in ordered.iterrows()
    ]
    ablation_rows = [
        (method, _row(summary, method)["rank"], _mean_std(_row(summary, method), "normalized_rmse_score"))
        for method in _ABLATION_METHODS
    ]
    hybrid = _row(summary, _HYBRID_METHOD)
    comparison_rows = []
    for reference_method in (_FIR_METHOD, "bias_drift"):
        reference = _row(summary, reference_method)
        comparison_rows.append((
            reference_method,
            hybrid["normalized_rmse_score_mean"] - reference["normalized_rmse_score_mean"],
            hybrid["gyro_rmse_mean"] - reference["gyro_rmse_mean"],
            hybrid["accel_rmse_mean"] - reference["accel_rmse_mean"],
        ))
    stability_rows = [
        (method, int(_row(summary, method)["seed_count"]), _mean_std(_row(summary, method), "normalized_rmse_score"))
        for method in _METHOD_ORDER
    ]
    spectral_rows = []
    for column, label, direction in _SPECTRAL_METRICS:
        best, worst = _best_worst(summary, column, direction)
        spectral_rows.append((label, "不可用" if best is None else best["method"], "不可用" if best is None else best["_value"], "不可用" if worst is None else worst["method"], "不可用" if worst is None else worst["_value"]))
    preintegration_rows = []
    for column, label in _PREINTEGRATION_METRICS:
        best, worst = _best_worst(summary, column, "min")
        preintegration_rows.append((label, "不可用" if best is None else best["method"], "不可用" if best is None else best["_value"], "不可用" if worst is None else worst["method"], "不可用" if worst is None else worst["_value"]))

    return "\n".join([
        "# ALiMU 严格跨传感器虚拟 IMU 报告",
        "",
        "## 实验范围",
        "",
        "本实验采用严格 leave_sensor_out: 训练和验证仅使用 BMX160、LSM9DS1, 测试仅使用 MPU9250; 目标为 40 Hz 的 VN100。",
        "固定矩阵包含 15 个神经网络试验 (5 个方法 x 3 个种子) 和 4 个解析基线。",
        "",
        "## 总体排名",
        "",
        "汇总按单一 heldout 传感器计算, 表中为均值 ± seed 标准差; 解析基线没有 seed 标准差。",
        "",
        *_table(("rank", "方法", "归一化 RMSE", "陀螺 RMSE", "加速度计 RMSE"), ranking_rows),
        "",
        "\u4e0d\u8ba1\u7b97 fold \u6807\u51c6\u5dee; \u4e0d\u786e\u5b9a\u6027\u4ec5\u4e3a seed \u6807\u51c6\u5dee\u3002",
        "",
        "## 四分支消融",
        "",
        *_table(("方法", "rank", "归一化 RMSE"), ablation_rows),
        "",
        "## 混合方法对比",
        "",
        "fir_bias_drift_residual 相对 FIR 和 bias_drift 的差值为前者减参考方法, 负值表示误差更低。",
        "",
        *_table(("参考方法", "归一化 RMSE 差", "陀螺 RMSE 差", "加速度计 RMSE 差"), comparison_rows),
        "",
        "## 种子稳定性",
        "",
        *_table(("方法", "seed 数", "归一化 RMSE (均值 ± seed 标准差)"), stability_rows),
        "",
        "## 频谱补充指标",
        "",
        "仅在输入汇总中存在且有限时报告最佳/最差方法; 缺失或非有限的补充指标标记为不可用。",
        "",
        *_table(("指标", "最佳方法", "最佳值", "最差方法", "最差值"), spectral_rows),
        "",
        "## 预积分补充指标",
        "",
        *_table(("指标", "最佳方法", "最佳值", "最差方法", "最差值"), preintegration_rows),
        "",
        "## 局限性",
        "",
        "ALiMU 是单一 heldout 传感器的严格跨传感器结果; 绝对 Penn 与 ALiMU RMSE 不可直接比较, 因为数据来源、传感器和划分协议不同。",
        "",
        "## 溯源",
        "",
        *[
            f"- {key}: {_format_provenance_value(provenance[key], (str(key),))}"
            for key in sorted(provenance, key=str)
        ],
        "",
    ])


def write_alimu_matrix_report(
    overall_summary: pd.DataFrame,
    trials: pd.DataFrame,
    provenance: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Render and write the ALiMU report, logging its artifact path."""
    report = render_alimu_matrix_report(overall_summary, trials, provenance)
    path = Path(output_path).resolve()
    run_dir = Path(provenance["run_dir"]).resolve()
    if not path.is_relative_to(run_dir):
        raise ValueError("output_path must be under provenance.run_dir")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    logger.info("WRITES kind=alimu_matrix_report path={}", path)
    return path
