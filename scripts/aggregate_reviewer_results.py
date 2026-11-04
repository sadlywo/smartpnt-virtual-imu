"""Aggregate completed reviewer artifacts without inference or new fitting.

Physical metrics preserve their units. Seed SD describes optimization variation,
and identical classical predictions are retained once with an explicit audit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import platform
import sys
import traceback
import uuid
from datetime import datetime
from importlib.metadata import version
from itertools import combinations
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.eval.formal_evaluation import HIGH, IDENTITY, LOW, physical_error_metrics
from virtual_imu.eval.formal_navigation import navigation_error_metrics
from virtual_imu.training.reviewer_hpo import (
    _verify_evaluation,
    _verify_selection,
    read_validation_score,
)

PAIRED_METHODS = ["model", "identity", "bias_mean", "affine", "ridge_affine", "learned_fir_causal", "arx_linear", "ridge_tuned", "fir_tuned", "arx_tuned"]
NAV_METHODS = ["model", "zero", "train_mean", "ridge", "ridge_tuned"]
NAV_AXES = ["delta_x_local_m", "delta_y_local_m", "delta_z_local_m"]
AXES = ["gx", "gy", "gz", "ax", "ay", "az"]
GROUPS = ["task", "protocol_id", "training_scope", "split", "aggregation_level", "dataset_id", "low_sensor", "high_sensor", "sequence_id", "original_sequence_id", "physical_group_id", "method", "model_family", "variant", "metric", "unit"]
DEFAULT_CONFIG = {"zero_identity_threshold": 1e-12, "metric_rtol": 1e-9, "metric_atol": 1e-11, "ranking_metrics": ["gyro_vector_rmse_radps", "accel_vector_rmse_mps2", "vector_rmse_m"]}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_json(value))


def _finite(value, where="metadata"):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{where}.{key}")
    elif isinstance(value, list):
        for item in value:
            _finite(item, where)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite {where}")


def _transfer_api():
    path = Path(__file__).with_name("evaluate_reviewer_transfer.py")
    spec = importlib.util.spec_from_file_location("reviewer_aggregation_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _processing_sources(project: Path, state: dict, *, allow_subset=False) -> dict:
    """Bind the current package roster to frozen hashes with portable identities."""
    project = project.resolve()
    package = project / "src" / "virtual_imu"
    executed = {path.relative_to(project).as_posix(): _sha(path) for path in sorted(package.rglob("*.py"))}
    if not executed:
        raise ValueError("Frozen src/virtual_imu package is missing from the working directory")
    frozen = {}
    for identity, expected in state.get("contract", {}).get("source_sha256", {}).items():
        parts = PurePosixPath(str(identity).replace("\\", "/")).parts
        anchors = [index for index in range(len(parts) - 1) if parts[index:index + 2] == ("src", "virtual_imu")]
        if not anchors:
            continue
        if len(anchors) != 1 or ".." in parts:
            raise ValueError("Frozen implementation source identity is ambiguous")
        relative = PurePosixPath(*parts[anchors[0]:]).as_posix()
        if relative in frozen and frozen[relative] != expected:
            raise ValueError("Conflicting frozen source hashes for one package identity")
        frozen[relative] = expected
        if executed.get(relative) != expected:
            raise ValueError(f"Executed source differs from frozen training source: {relative}")
    if not allow_subset and (not frozen or set(executed) != set(frozen)):
        raise ValueError("Executed/frozen implementation source rosters differ")
    return {"project_root": str(project), "strict_roster_verified": not allow_subset, "matched_trained_source_sha256": frozen, "executed_source_sha256": executed}


def training_scope(protocol: str) -> str:
    if protocol == "pooled":
        return "pooled_alimu_penn"
    if protocol.startswith("penn_loso_"):
        return "penn_only_outer_recording"
    if protocol.startswith("navigation_rotation_"):
        return "smartpnt_outer_trajectory"
    return protocol


def metric_unit(metric: str) -> str:
    if metric.endswith("radps"):
        return "rad/s"
    if metric.endswith("mps2"):
        return "m/s^2"
    if metric.endswith("_m"):
        return "m"
    raise ValueError(f"Metric has no explicitly supported physical unit: {metric}")


def physical_metrics(task: str, prediction: np.ndarray, reference: np.ndarray) -> dict:
    """Flatten exactly the scalar/axis metrics emitted by the core evaluator."""
    if task == "paired":
        raw = physical_error_metrics(prediction, reference)
        error = np.asarray(prediction, float) - np.asarray(reference, float)
        for name, axis, unit in (("gyro", slice(0, 3), "radps"), ("accel", slice(3, 6), "mps2")):
            raw[f"{name}_vector_mae_{unit}"] = float(np.linalg.norm(error[:, axis], axis=1).mean())
            raw[f"{name}_bias_norm_{unit}"] = float(np.linalg.norm(error[:, axis].mean(axis=0)))
        result = {key: value for key, value in raw.items() if np.isscalar(value)}
        for index, axis in enumerate(AXES):
            for key, label in (("axis_rmse", "rmse"), ("axis_mae", "mae"), ("axis_mean_error", "mean_error"), ("axis_error_std", "error_std")):
                result[f"{axis}_{label}_{'radps' if index < 3 else 'mps2'}"] = raw[key][index]
        return result
    raw = navigation_error_metrics(prediction, reference)
    error = np.asarray(prediction, float) - np.asarray(reference, float)
    raw["vector_mae_m"] = float(np.linalg.norm(error, axis=1).mean())
    raw["axis_error_std_m"] = error.std(axis=0).tolist()
    raw["bias_norm_m"] = float(np.linalg.norm(error.mean(axis=0)))
    result = {key: value for key, value in raw.items() if np.isscalar(value)}
    for index, axis in enumerate(("east", "north", "up")):
        for key, label in (("axis_rmse_m", "rmse"), ("axis_mae_m", "mae"), ("axis_bias_m", "bias"), ("axis_error_std_m", "error_std")):
            result[f"{axis}_{label}_m"] = raw[key][index]
    return result


def _frame_digest(frame: pd.DataFrame, columns: list[str], order: list[str]) -> str:
    selected = frame.sort_values(order, kind="stable")[columns].reset_index(drop=True)
    digest = hashlib.sha256(json.dumps(columns).encode())
    digest.update(pd.util.hash_pandas_object(selected, index=False).to_numpy().tobytes())
    return digest.hexdigest()


class ClassicalAudit:
    """Deduplicate only bit-identical physical coverage, outputs and CSV metrics."""

    def __init__(self):
        self.records = {}

    def observe(self, key: tuple, *, physical: str, prediction: str, metrics: str, job_id: str) -> bool:
        fingerprint = {"physical_rows_sha256": physical, "predictions_sha256": prediction, "saved_metrics_sha256": metrics}
        if key in self.records:
            record = self.records[key]
            if any(record[name] != value for name, value in fingerprint.items()):
                raise ValueError(f"Repeated classical coverage, predictions or metrics differ: {key}")
            record["source_job_ids"].append(job_id)
            return False
        self.records[key] = {"task": key[0], "protocol_id": key[1], "split": key[2], "method": key[3], **fingerprint, "representative_job_id": job_id, "source_job_ids": [job_id]}
        return True


def _enrich(frame: pd.DataFrame, task: str, catalog: dict) -> pd.DataFrame:
    frame = frame.copy()
    index = {record["sequence_id"]: record for record in catalog.get("records", [])}
    for column in ("dataset_id", "low_sensor", "high_sensor", "original_sequence_id", "physical_group_id"):
        values = frame.sequence_id.map(lambda identity, name=column: index.get(identity, {}).get(name))
        if column not in frame:
            if values.isna().any():
                raise ValueError(f"Saved predictions/catalog lack physical identity {column}")
            frame[column] = values
        elif values.notna().any() and not frame.loc[values.notna(), column].astype(str).eq(values.loc[values.notna()].astype(str)).all():
            raise ValueError(f"Saved prediction identity differs from the frozen catalog: {column}")
    if frame[["sequence_id", *IDENTITY, "original_sequence_id", "physical_group_id"]].isna().any().any():
        raise ValueError("Saved prediction physical identities cannot be missing")
    if task == "paired":
        if frame.duplicated(["sequence_id", "timestamp_s"]).any() or not np.isfinite(frame[LOW + HIGH + ["timestamp_s"]].to_numpy(float)).all():
            raise ValueError("Paired predictions require unique finite physical timestamps")
        for _, group in frame.groupby("sequence_id", sort=False):
            if len(group[IDENTITY].drop_duplicates()) != 1 or not np.allclose(np.diff(np.sort(group.timestamp_s)), 0.025, atol=1e-6, rtol=1e-5):
                raise ValueError("Paired continuous-sequence identity or sampling is invalid")
    else:
        keys = ["sequence_id", "start_index", "end_index"]
        if frame.duplicated(keys).any() or not np.allclose(frame.end_time_s - frame.start_time_s, 1.975, atol=1e-6, rtol=1e-5) or not (frame.end_index - frame.start_index).eq(79).all():
            raise ValueError("Navigation rows must be unique 1.975s displacement windows")
        for _, group in frame.groupby("sequence_id", sort=False):
            if not np.all(np.diff(np.sort(group.start_index)) == 40):
                raise ValueError("Navigation windows violate within-segment stride40 support")
    return frame


def _verify_saved(computed: dict, saved: pd.Series, config: dict):
    for metric, value in computed.items():
        if metric not in saved or not np.isfinite(saved[metric]) or not math.isclose(float(saved[metric]), float(value), rel_tol=config["metric_rtol"], abs_tol=config["metric_atol"]):
            raise ValueError(f"Saved physical CSV metric disagrees with prediction rows: {metric}")


def prediction_metrics(frame: pd.DataFrame, saved: pd.DataFrame, entry: dict, task: str, split: str, catalog: dict, classical: ClassicalAudit, coverage: dict, config: dict) -> list[dict]:
    frame = _enrich(frame, task, catalog)
    methods = PAIRED_METHODS if task == "paired" else NAV_METHODS
    target = HIGH if task == "paired" else [f"true_{axis}" for axis in NAV_AXES]
    order = ["sequence_id", "timestamp_s"] if task == "paired" else ["sequence_id", "start_index", "end_index"]
    physical = [*IDENTITY, "sequence_id", "original_sequence_id", "physical_group_id", *(["timestamp_s", *LOW, *HIGH] if task == "paired" else ["start_index", "end_index", "start_time_s", "end_time_s", *target])]
    physical_sha = _frame_digest(frame, physical, order)
    coverage_key = (task, entry["protocol_id"], split)
    if coverage_key in coverage and coverage[coverage_key] != physical_sha:
        raise ValueError("Model contexts/seeds do not share complete physical-row coverage")
    coverage[coverage_key] = physical_sha
    selected_csv = saved.loc[saved.split.eq(split)].copy()
    if set(selected_csv.method) != set(methods):
        raise ValueError("Saved evaluation CSV does not contain every declared method")
    result = []
    for method in methods:
        prefix = method if task == "paired" or method != "model" else "pred"
        predicted = [f"{prefix}_{axis}" for axis in (AXES if task == "paired" else NAV_AXES)]
        if not np.isfinite(frame[predicted + target].to_numpy(float)).all():
            raise ValueError("Saved predictions/targets must be finite")
        csv_rows = selected_csv.loc[selected_csv.method.eq(method)].sort_values(IDENTITY if task == "paired" else ["method"]).reset_index(drop=True)
        if method != "model":
            prediction_sha = _frame_digest(frame, physical + predicted, order)
            metric_sha = _frame_digest(csv_rows, list(csv_rows.columns), IDENTITY if task == "paired" else ["method"])
            if not classical.observe((task, entry["protocol_id"], split, method), physical=physical_sha, prediction=prediction_sha, metrics=metric_sha, job_id=entry["job_id"]):
                continue
        groups = [("dataset", ["dataset_id"]), ("pair", IDENTITY), ("sequence", [*IDENTITY, "sequence_id"])]
        for level, keys in groups:
            for _, group in frame.groupby(keys, sort=True):
                values = physical_metrics(task, group[predicted].to_numpy(float), group[target].to_numpy(float))
                if (task == "paired" and level == "pair") or (task == "navigation" and level == "dataset"):
                    matching = csv_rows
                    if task == "paired":
                        for name in IDENTITY:
                            matching = matching.loc[matching[name].astype(str).eq(str(group[name].iloc[0]))]
                    if len(matching) != 1:
                        raise ValueError("Saved physical CSV has ambiguous or missing identity")
                    _verify_saved(values, matching.iloc[0], config)
                def single(name, selected=group):
                    return str(selected[name].iloc[0]) if selected[name].nunique() == 1 else "__multiple__"
                identity = {"dataset_id": str(group.dataset_id.iloc[0]), "low_sensor": single("low_sensor") if level != "dataset" else "__all_devices__", "high_sensor": single("high_sensor") if level != "dataset" else "__all_devices__", "sequence_id": str(group.sequence_id.iloc[0]) if level == "sequence" else "__all_sequences__", "original_sequence_id": single("original_sequence_id") if level == "sequence" else "__multiple__", "physical_group_id": single("physical_group_id") if level == "sequence" else "__multiple__"}
                base = {"task": task, "protocol_id": entry["protocol_id"], "training_scope": training_scope(entry["protocol_id"]), "split": split, "aggregation_level": level, **identity, "method": method, "model_family": entry["model_family"] if method == "model" else method, "variant": entry["variant"] if method == "model" else "classical", "seed": entry["seed"] if method == "model" else None, "source_job_id": entry["job_id"], "support_count": len(group), "support_unit": "timestamps" if task == "paired" else "windows", "physical_recording_count": group.physical_group_id.nunique()}
                for metric, value in values.items():
                    if metric not in {"samples", "windows"}:
                        result.append({**base, "metric": metric, "unit": metric_unit(metric), "value": float(value)})
    return result


def summarize_long(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    duplicated = frame.duplicated([*GROUPS, "seed"])
    if duplicated.any():
        raise ValueError("Repeated metric/optimization-seed identity")
    result = frame.groupby(GROUPS, dropna=False, as_index=False).agg(mean=("value", "mean"), sample_sd=("value", "std"), n=("value", "count"), optimization_seed_count=("seed", "nunique"), support_count=("support_count", "first"), physical_recording_count=("physical_recording_count", "first"))
    if not result.loc[result.variant.eq("classical"), "n"].eq(1).all():
        raise ValueError("Classical summaries must represent one verified deterministic prediction result")
    return result


def paired_deltas(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = long.loc[long.method.eq("model")]
    keys = [key for key in GROUPS if key != "variant"] + ["seed"]
    default = model.loc[model.variant.eq("default")]
    tuned = model.loc[model.variant.eq("tuned")]
    paired = default.merge(tuned, on=keys, suffixes=("_default", "_tuned"), validate="one_to_one")
    if len(paired) != len(default) or len(paired) != len(tuned):
        raise ValueError("Default/tuned metrics lack exact matched seed/target identities")
    paired["delta_tuned_minus_default"] = paired.value_tuned - paired.value_default
    positive_error = ~paired.metric.str.contains(r"mean_error|^(?:east|north|up)_bias", regex=True)
    paired["percent_change"] = 100 * np.divide(paired.delta_tuned_minus_default, paired.value_default, out=np.full(len(paired), np.nan), where=(paired.value_default > 1e-12) & positive_error)
    summary = paired.groupby([key for key in keys if key != "seed"], dropna=False, as_index=False).agg(delta_mean=("delta_tuned_minus_default", "mean"), delta_sample_sd=("delta_tuned_minus_default", "std"), paired_seed_count=("seed", "nunique"), percent_change_mean=("percent_change", "mean"))
    return paired, summary


def relative_identity(long: pd.DataFrame, threshold: float) -> pd.DataFrame:
    selected = long.loc[long.task.eq("paired") & long.aggregation_level.eq("pair") & long.metric.isin(["gyro_vector_rmse_radps", "accel_vector_rmse_mps2"])].copy()
    keys = ["protocol_id", "split", *IDENTITY, "metric"]
    identity = selected.loc[selected.method.eq("identity"), [*keys, "value"]].rename(columns={"value": "identity_value"})
    ratios = selected.merge(identity, on=keys, validate="many_to_one")
    ratios["ratio"] = np.divide(ratios.value, ratios.identity_value, out=np.full(len(ratios), np.nan), where=ratios.identity_value > threshold)
    per_dataset = ["protocol_id", "training_scope", "split", "method", "model_family", "variant", "seed", "metric", "dataset_id"]
    values = ratios.groupby(per_dataset, dropna=False, as_index=False).agg(ratio=("ratio", "mean"), pairs=("ratio", "count"), excluded=("ratio", lambda group: int(group.isna().sum())))
    result = values.groupby(per_dataset[:-1], dropna=False, as_index=False).agg(relative_rmse_percent=("ratio", lambda group: 100 * group.mean()), included_pairs=("pairs", "sum"), excluded_near_zero_identity_pairs=("excluded", "sum"), datasets=("dataset_id", "nunique"))
    result["improvement_percent_vs_identity"] = 100 - result.relative_rmse_percent
    result["unit"] = "%"
    return result


def ranking_tables(summary: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranks = summary.loc[summary.method.eq("model") & summary.aggregation_level.isin(["dataset", "pair"]) & summary.metric.isin(metrics)].copy()
    domains = ["task", "protocol_id", "training_scope", "split", "aggregation_level", *IDENTITY, "variant", "metric", "unit"]
    ranks["rank"] = ranks.groupby(domains, dropna=False)["mean"].rank(method="average")
    correlations = []
    for (task, split, level, variant, metric, unit), frame in ranks.groupby(["task", "split", "aggregation_level", "variant", "metric", "unit"], sort=True):
        group_keys = ["protocol_id", *IDENTITY]
        grouped = list(frame.groupby(group_keys, sort=True))
        for (left, a), (right, b) in combinations(grouped, 2):
            category = None
            if left[0] == right[0] == "pooled":
                category = "pooled_training_evaluation_slices"
            elif {left[0], right[0]} == {"alimu_only", "penn_only"}:
                category = "different_training_and_evaluation_datasets"
            elif {left[0], right[0]} in ({"pooled", "alimu_only"}, {"pooled", "penn_only"}) and left[1:] == right[1:]:
                category = "same_target_different_training_scope"
            if category is None:
                continue
            matched = a[["model_family", "mean"]].merge(b[["model_family", "mean"]], on="model_family", suffixes=("_left", "_right"), validate="one_to_one")
            if len(matched) < 2:
                continue
            rho = spearmanr(matched.mean_left, matched.mean_right).statistic
            tau = kendalltau(matched.mean_left, matched.mean_right).statistic
            correlations.append({"task": task, "split": split, "aggregation_level": level, "variant": variant, "metric": metric, "unit": unit, "comparison": category, "left_protocol": left[0], "left_dataset": left[1], "left_low_sensor": left[2], "left_high_sensor": left[3], "right_protocol": right[0], "right_dataset": right[1], "right_low_sensor": right[2], "right_high_sensor": right[3], "models": len(matched), "spearman_rho": float(rho) if np.isfinite(rho) else None, "kendall_tau_b": float(tau) if np.isfinite(tau) else None})
    correlation_columns = ["task", "split", "aggregation_level", "variant", "metric", "unit", "comparison", "left_protocol", "left_dataset", "left_low_sensor", "left_high_sensor", "right_protocol", "right_dataset", "right_low_sensor", "right_high_sensor", "models", "spearman_rho", "kendall_tau_b"]
    return ranks, pd.DataFrame(correlations, columns=correlation_columns)


def _elapsed(job: dict):
    if not job.get("started_utc") or not job.get("completed_utc"):
        return None
    value = (datetime.fromisoformat(job["completed_utc"]) - datetime.fromisoformat(job["started_utc"])).total_seconds()
    if value < 0:
        raise ValueError("Persisted job timing runs backwards")
    return value


def _report(summary: pd.DataFrame, root: Path, metrics: list[str]) -> str:
    lines = ["# Reviewer result aggregation", "", f"Source core: `{root}`.", "", "Validation and test remain separate. Neural rows show mean ± sample SD across three optimization seeds; these are not independent recording replicates, confidence intervals or p-values. Classical rows have n=1 after complete physical-row/output/metric equality checks, so their SD is blank.", "", "Gyroscope rad/s, accelerometer m/s² and navigation displacement m remain separate. Navigation evaluates 1.975s ENU displacement windows; no trajectory is assembled. Continuous sequences are scored independently; temporal diagnostics are not recomputed across segments.", ""]
    primary = summary.loc[summary.aggregation_level.isin(["dataset", "pair"]) & summary.metric.isin(metrics)]
    for identity, frame in primary.groupby(["split", "protocol_id", "aggregation_level", *IDENTITY, "metric", "unit"], sort=True):
        split, protocol, level, dataset, low, high, metric, unit = identity
        lines.extend([f"## {split} · {protocol} · {level} · {dataset} · {low} → {high} · {metric} [{unit}]", "", "| Method | Variant | Mean | Sample SD | n |", "|---|---|---:|---:|---:|"])
        for row in frame.sort_values("mean").itertuples():
            sd = "" if pd.isna(row.sample_sd) else f"{row.sample_sd:.6g}"
            lines.append(f"| {row.model_family} | {row.variant} | {row.mean:.6g} | {sd} | {row.n} |")
        lines.append("")
    lines.extend(["Rank correlations are descriptive. Pooled-training domain slices, separately trained domains, and same-target changes of training scope have distinct labels. Default/tuned deltas pair the same seed and held-out physical rows; negative error deltas indicate lower tuned error. No winner is guaranteed.", "", "Relative-identity percentages average device-pair RMSE ratios within each dataset, then average datasets equally. Improvement percent equals 100 minus this ratio percentage: positive means lower RMSE than identity, negative means higher RMSE. Gyroscope and accelerometer ratios remain separate. Near-zero identity denominators are flagged and excluded, with counts retained.", "", "Fit elapsed time comes from persisted start/completion timestamps and includes the training backend plus checkpoint verification. Failed and incomplete fits remain in the job ledger. SmartPNT rotations share trajectories across roles and do not supply independent environment replications.", ""])
    return "\n".join(lines)


def verify_summary(out: Path, root: Path, config: dict) -> dict:
    manifest = json.loads((out / "artifact_manifest.json").read_text(encoding="utf-8"))
    if manifest["pipeline_root"] != str(root.resolve()) or manifest["config"] != config or manifest["script_sha256"] != _sha(Path(__file__)) or manifest["gate_script_sha256"] != _sha(Path(__file__).with_name("evaluate_reviewer_transfer.py")):
        raise ValueError("Aggregation source/config/implementation identity changed")
    for name, expected in manifest["files"].items():
        if _sha(out / name) != expected:
            raise ValueError(f"Summary output artifact changed: {name}")
    for name, expected in manifest["source_artifacts"].items():
        if _sha(Path(name)) != expected:
            raise ValueError(f"Summary source artifact changed: {name}")
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


def aggregate_results(root: Path, out: Path, *, config: dict | None = None, resume=False, allow_subset=False) -> dict:
    config = {**DEFAULT_CONFIG, **(config or {})}
    root, out = root.resolve(), out.resolve()
    if out == root or out.is_relative_to(root / "experiments"):
        raise ValueError("Summary output must not modify core checkpoint directories")
    state_path = root / "pipeline.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _finite(state, "pipeline")
    gate = _transfer_api()
    finals = gate.verify_pipeline(state, root, allow_subset=allow_subset)
    if state.get("status") != "complete" or state.get("smoke"):
        raise ValueError("Aggregation requires a completed nonsmoke core")
    processing_sources = _processing_sources(Path.cwd(), state, allow_subset=allow_subset)
    if out.exists():
        if not resume:
            raise FileExistsError(f"Preserving existing aggregation: {out}")
        return verify_summary(out, root, config)
    if resume:
        raise FileNotFoundError("No completed aggregation exists to resume")
    print(f"READS: completed core, saved checkpoint validation and prediction artifacts {root}", flush=True)
    print(f"WRITES: independent metric/ranking tables and source manifests {out}", flush=True)
    print("SKIPS: inference, fitting, test-based model selection, trajectory stitching, mixed-unit scores and inferential p-values", flush=True)
    sources = {str(state_path): _sha(state_path), str(root / "matrix_config.yaml"): _sha(root / "matrix_config.yaml")}
    for relative, expected in processing_sources["executed_source_sha256"].items():
        sources[str(Path(processing_sources["project_root"]) / relative)] = expected
    for function in (physical_error_metrics, navigation_error_metrics, _verify_selection, _verify_evaluation, read_validation_score):
        path = Path(inspect.getfile(function)).resolve()
        sources[str(path)] = _sha(path)
    if yaml.safe_load((root / "matrix_config.yaml").read_text(encoding="utf-8")) != state["contract"]["config"]:
        raise ValueError("Frozen matrix config differs from the source contract")
    matrix = state["contract"]["config"]
    declared = {f"{protocol['id']}/{candidate['id']}" for protocol in matrix["protocols"] for candidate in matrix["candidates"] if protocol["task"] in candidate["tasks"]}
    if set(state["cells"]) != declared or set(state["contract"]["cells"]) != declared or state["final_seeds"] != matrix["seeds"]:
        raise ValueError("Completed model/protocol/seed roster differs from the declared matrix")
    for cell_id in state["cells"]:
        _verify_selection(cell_id, state, root)
        path = root / "selections" / f"{cell_id.replace('/', '__')}.json"
        sources[str(path)] = _sha(path)
    staging = out.parent / f".{out.name}.{uuid.uuid4().hex}.partial"
    staging.mkdir(parents=True)
    long_rows, registry, job_rows = [], [], []
    classical, coverage = ClassicalAudit(), {}
    try:
        for job in state["jobs"].values():
            row = {key: job.get(key) for key in ("id", "cell", "phase", "trial_id", "rung", "seed", "epochs", "status", "score", "run_dir", "config", "config_sha256", "error_type", "error", "failure_stage", "failure_recoverable", "evaluation_status", "started_utc", "completed_utc")}
            path = Path(job["config"])
            if _sha(path) != job["config_sha256"]:
                raise ValueError("Persisted fit config checksum changed")
            sources[str(path)] = job["config_sha256"]
            fit_config = yaml.safe_load(path.read_text(encoding="utf-8"))
            row.update(best_epoch=job.get("validation", {}).get("best_epoch"), validation_criterion=fit_config["training"]["checkpoint_metric"], actual_epochs=None, early_stopped=None)
            if job["status"] == "complete":
                history_path = Path(job["run_dir"]) / "metrics.csv"
                expected = job["artifact_sha256"]["metrics.csv"]
                if _sha(history_path) != expected:
                    raise ValueError("Saved fit history checksum changed")
                sources[str(history_path)] = expected
                fit_history = pd.read_csv(history_path)
                selected_values = fit_history[row["validation_criterion"]].to_numpy(float)
                if not len(fit_history) or not np.isfinite(selected_values).all() or not math.isclose(float(selected_values.min()), job["score"], rel_tol=1e-6, abs_tol=1e-9):
                    raise ValueError("Saved fit validation history disagrees with the selection ledger")
                row["actual_epochs"] = int(fit_history.epoch.max())
                row["early_stopped"] = row["actual_epochs"] < job["epochs"]
            row["elapsed_seconds"] = _elapsed(job)
            row["elapsed_scope"] = "persisted backend plus checkpoint-verification wall time"
            job_rows.append(row)
        for job in sorted(finals, key=lambda entry: entry["id"]):
            directory = Path(job["run_dir"]).resolve()
            if not directory.is_relative_to(root):
                raise ValueError("Core run paths must resolve inside the supplied pipeline")
            frozen = Path(job["config"])
            if _sha(frozen) != job["config_sha256"]:
                raise ValueError("Frozen final config checksum changed")
            actual = yaml.safe_load(frozen.read_text(encoding="utf-8"))
            _finite(actual, "config")
            for name, expected in job["artifact_sha256"].items():
                path = directory / name
                if _sha(path) != expected:
                    raise ValueError(f"Final training artifact checksum changed: {path}")
                sources[str(path)] = expected
            evidence = read_validation_score(directory, actual)
            if not math.isclose(evidence["score"], job["score"], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Final validation score differs from pipeline state")
            _verify_evaluation(job, root)
            for path, expected in job["evaluation_artifact_sha256"].items():
                sources[path] = expected
            sources[str(frozen)] = job["config_sha256"]
            cell = state["cells"][job["cell"]]
            metadata = actual["reviewer_protocol"]
            variant = job["phase"].removeprefix("final_")
            if metadata["protocol_id"] != cell["protocol"]["id"] or metadata["model_family"] != cell["candidate"]["id"] or metadata["variant"] != variant or actual["seed"] != job["seed"]:
                raise ValueError("Final model/protocol/variant/seed identity differs")
            expected_trial = cell["trials"][0] if variant == "default" else cell["selected"]["trial"]
            if metadata["parameters"] != expected_trial["parameters"] or job["trial_id"] != expected_trial["id"]:
                raise ValueError("Final default/selected hyperparameters differ from frozen selection")
            history = pd.read_csv(directory / "metrics.csv")
            actual_epochs = int(history.epoch.max())
            entry = {"job_id": job["id"], "task": cell["protocol"]["task"], "protocol_id": metadata["protocol_id"], "training_scope": training_scope(metadata["protocol_id"]), "model_family": metadata["model_family"], "variant": variant, "seed": job["seed"], "best_epoch": evidence["best_epoch"], "actual_epochs": actual_epochs, "epoch_budget": actual["training"]["epochs"], "early_stopped": actual_epochs < actual["training"]["epochs"], "validation_criterion": evidence["metric"], "validation_best_metric": evidence["score"], "elapsed_seconds": _elapsed(job), "window_points": actual["data"]["window_size"], "run_dir": str(directory), "config": str(frozen), "config_sha256": job["config_sha256"], "checkpoint_sha256": job["checkpoint_sha256"]}
            registry.append(entry)
            catalog = json.loads((directory / "data_manifest.json").read_text(encoding="utf-8"))
            saved = pd.read_csv(directory / "evaluation" / f"{entry['task']}_metrics.csv")
            if set(saved.split) != {"validation", "test"}:
                raise ValueError("Saved metrics must keep validation and test explicitly separate")
            for split in ("validation", "test"):
                frame = pd.read_parquet(directory / "evaluation" / f"{split}_predictions.parquet")
                long_rows.extend(prediction_metrics(frame, saved, entry, entry["task"], split, catalog, classical, coverage, config))
        long = pd.DataFrame(long_rows)
        summary = summarize_long(long)
        if not summary.loc[summary.method.eq("model"), "optimization_seed_count"].eq(3).all():
            raise ValueError("Every neural metric requires all three optimization seeds")
        deltas, delta_summary = paired_deltas(long)
        relative = relative_identity(long, config["zero_identity_threshold"])
        relative_keys = ["protocol_id", "training_scope", "split", "method", "model_family", "variant", "metric", "unit"]
        relative_summary = relative.groupby(relative_keys, dropna=False, as_index=False).agg(mean_percent=("relative_rmse_percent", "mean"), sample_sd_percent=("relative_rmse_percent", "std"), mean_improvement_percent_vs_identity=("improvement_percent_vs_identity", "mean"), sample_sd_improvement_percent=("improvement_percent_vs_identity", "std"), n=("relative_rmse_percent", "count"), optimization_seed_count=("seed", "nunique"), included_pairs=("included_pairs", "first"), excluded_near_zero_identity_pairs=("excluded_near_zero_identity_pairs", "first"))
        ranks, correlations = ranking_tables(summary, config["ranking_metrics"])
        tables = {"run_registry.csv": pd.DataFrame(registry), "all_fit_jobs.csv": pd.DataFrame(job_rows), "metrics_long.csv": long, "metrics_seed_summary.csv": summary, "matched_seed_deltas.csv": deltas, "matched_seed_delta_summary.csv": delta_summary, "relative_identity_percent.csv": relative, "relative_identity_seed_summary.csv": relative_summary, "model_rankings.csv": ranks, "rank_correlations.csv": correlations}
        for name, frame in tables.items():
            frame.to_csv(staging / name, index=False)
            print(f"WRITES: {name} rows={len(frame)}", flush=True)
        _write_json(staging / "classical_deduplication.json", list(classical.records.values()))
        (staging / "comparison.md").write_text(_report(summary, root, config["ranking_metrics"]), encoding="utf-8")
        (staging / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        report = {"status": "complete", "pipeline_root": str(root), "git_commit": state.get("git_commit"), "runs": len(registry), "fit_jobs": len(job_rows), "failed_fit_jobs": sum(row["status"] == "failed" for row in job_rows), "metric_rows": len(long), "classical_unique_protocol_split_methods": len(classical.records), "source_pipeline_sha256": sources[str(state_path)], "analysis_environment": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "pandas", "scipy", "torch")}}, "no_inference_or_fitting": True, "statistics_scope": "optimization-seed mean/sample SD only; no recording CI or p-values", "units": {"gyro": "rad/s", "accel": "m/s^2", "navigation": "m"}, "navigation_scope": "1.975-second local ENU endpoint displacement, no concatenated trajectory"}
        _write_json(staging / "summary.json", report)
        if _sha(state_path) != sources[str(state_path)]:
            raise ValueError("Core pipeline changed during aggregation")
        files = {path.relative_to(staging).as_posix(): _sha(path) for path in staging.rglob("*") if path.is_file()}
        _write_json(staging / "artifact_manifest.json", {"schema_version": 1, "pipeline_root": str(root), "config": config, "script_sha256": _sha(Path(__file__)), "gate_script_sha256": _sha(Path(__file__).with_name("evaluate_reviewer_transfer.py")), "processing_sources": processing_sources, "source_artifacts": sources, "files": files})
        staging.rename(out)
        return report
    except Exception as error:
        _write_json(staging / "failure.json", {"error": repr(error), "traceback": traceback.format_exc(), "partial_artifacts_preserved": True})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/reviewer_result_summary.yaml")
    args = parser.parse_args(argv)
    print(f"READS: result-summary YAML {args.config}", flush=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = aggregate_results(args.pipeline_root, args.out, config=config, resume=args.resume)
    print(f"WRITES: complete summary {args.out} runs={result['runs']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
