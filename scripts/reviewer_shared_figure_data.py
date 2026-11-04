"""Saved-result adapters for scope-separated shared reviewer figures.

No fitting, model loading, inference or dataset decoding belongs in this module.
The report consumer supplies the common strict public-interface validator; its
source hash is part of every shared figure build contract.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

PROFILE = "reviewer_shared_classical_v1"
SCOPE = "protocol_shared_classical"
BANK = "known_device_bank"
METHODS = [
    "identity",
    "bias_mean",
    "affine",
    "ridge_affine",
    "learned_fir_causal",
    "arx_linear",
    "ridge_tuned",
    "fir_tuned",
    "arx_tuned",
]
AXES = ["gx", "gy", "gz", "ax", "ay", "az"]
PHYSICAL_KEYS = ["dataset_id", "low_sensor", "high_sensor", "sequence_id", "timestamp_s"]
COUNTS = {
    "shared_protocols": 6,
    "shared_logical_controls": 54,
    "shared_learned_controls": 48,
    "shared_identity_controls": 6,
    "shared_signal_slots": 108,
    "shared_downstream_views": 5,
    "shared_downstream_summary_slots": 360,
    "shared_paired_final_mappings": 360,
}


def analysis_profile(package):
    """Reject unknown/mismatched profiles before any result payload is read."""
    outer = package.manifest.get("analysis_profile")
    inner = package.read_json("metadata/completion_audit.json").get("analysis_profile")
    if outer is None and inner is None:
        return None
    if outer != PROFILE or inner != PROFILE:
        raise ValueError("Unknown or mismatched reviewer analysis_profile")
    if not {"aux3_project", "aux3"} <= set(package.roots):
        raise ValueError("Shared analysis profile requires exact aux3 project/run roots")
    return PROFILE


def load_shared(reader, roots, finals, core_sha):
    """Use the same semantic public-artifact gate as the report consumer."""
    from build_reviewer_report import _shared_audit

    result = _shared_audit(reader.package, {job["id"]: job for job in finals}, core_sha)
    if result["root"] != roots["aux3"]:
        raise ValueError("Shared index exact aux3 root differs")
    receipt = reader.json("metadata/completion_audit.json").get("shared_controls_index", {})
    if (
        receipt.get("sha256") != result["index_sha256"]
        or not receipt.get("source_path")
        or reader.package.resolve_path(receipt["source_path"])
        != roots["aux3"] / "shared_controls_artifacts.json"
    ):
        raise ValueError("Completion audit shared index exact path/checksum differs")
    # Include every verified member, including manifest/source/dependency proofs,
    # in the figure source index. Never recover an artifact through latest/.
    reader.sources.update(reader.package.verified_files)
    tables = {}
    for role, descriptor in result["descriptors"].items():
        path = roots["aux3"] / descriptor["path"]
        frame = reader.csv(path)
        if len(frame) != descriptor["rows"] or list(frame) != descriptor["columns"]:
            raise ValueError("Shared figure table schema differs: " + role)
        tables[role] = frame
    result["frames"] = tables
    return result


def _deterministic(frame):
    if not frame.n.eq(1).all() or frame.sd.notna().any():
        raise ValueError("Shared figure statistics require n=1 and blank SD")


def metric_unit(metric):
    if metric.endswith("relative_identity") or metric == "grouped_relative_score":
        return "dimensionless"
    if metric == "samples":
        return "samples"
    if metric.startswith("gyro") or metric.endswith(("_gx", "_gy", "_gz")):
        return "rad/s"
    if metric.startswith("accel") or metric.endswith(("_ax", "_ay", "_az")):
        return "m/s^2"
    raise ValueError("Shared physical metric has no declared unit: " + metric)


def main_metrics(original, shared):
    """Join saved neural statistics and shared n=1 rows without averaging scopes."""
    required = {
        "protocol",
        "information_scope",
        "canonical_method",
        "variant",
        "split",
        "metric_scope",
        "metric",
        "value",
        "n",
        "sd",
    }
    if required - set(shared):
        raise ValueError("Shared metrics schema missing scope/statistical identity")
    if set(shared.information_scope) != {SCOPE} or set(shared.variant) != {"shared"}:
        raise ValueError("Shared metrics contain a different information scope/variant")
    _deterministic(shared)
    physical = shared[shared.metric_scope.eq("pair")].copy()
    physical = physical.rename(
        columns={
            "protocol": "protocol_id",
            "metric_scope": "aggregation_level",
            "canonical_method": "method",
            "value": "mean",
            "sd": "sample_sd",
        }
    )
    physical["task"] = "paired"
    physical["model_family"] = physical.method
    physical["unit"] = physical.metric.map(metric_unit)
    keys = [
        "protocol_id",
        "split",
        "dataset_id",
        "low_sensor",
        "high_sensor",
        "method",
        "variant",
        "metric",
    ]
    if physical.duplicated(keys).any():
        raise ValueError("Shared metrics repeat a physical method/statistical unit")
    neural = original[original.task.eq("paired") & original.method.eq("model")].copy()
    neural["information_scope"] = "protocol_unified_neural"
    # Each source pair/split/metric must be the exact same comparison unit.
    support = ["protocol_id", "split", "dataset_id", "low_sensor", "high_sensor", "metric"]
    basic = [
        "gyro_vector_rmse_radps",
        "gyro_vector_mae_radps",
        "gyro_bias_norm_radps",
        "accel_vector_rmse_mps2",
        "accel_vector_mae_mps2",
        "accel_bias_norm_mps2",
    ]
    left = neural[neural.aggregation_level.eq("pair") & neural.metric.isin(basic)][
        support
    ].drop_duplicates()
    right = physical[physical.metric.isin(basic)][support].drop_duplicates()
    if set(map(tuple, left.to_numpy())) != set(map(tuple, right.to_numpy())):
        raise ValueError("Shared/neural physical metric support differs")
    identity = physical[physical.method.eq("identity")]
    if identity.duplicated(support).any():
        raise ValueError("Common identity must occur once per comparison unit")
    return pd.concat([neural, physical], ignore_index=True)


def main_downstream(original, shared):
    """Preserve three-seed neural means/SD and each saved deterministic outcome."""
    if set(shared.information_scope) != {SCOPE} or set(shared.variant) != {"shared"}:
        raise ValueError("Shared downstream scope/variant differs")
    _deterministic(shared)
    shared = shared.copy()
    shared["method"] = shared.canonical_method
    for stem in ("endpoint_rmse_deg", "position_endpoint_rmse_m"):
        shared[stem + "_mean"] = shared[stem]
        shared[stem + "_sample_sd"] = np.nan
        active = shared.task.eq("attitude" if stem.startswith("endpoint") else "strapdown")
        missing = active & ~np.isfinite(shared[stem].to_numpy(float))
        if (
            missing.any()
            and not shared.loc[missing, "status"].astype(str).str.startswith("unsupported_").all()
        ):
            raise ValueError("Nonfinite downstream outcome lacks explicit unsupported status")
    retain = original.method.str.contains("/", regex=False) | original.method.eq("HIGH")
    neural = original[retain].copy()
    neural["information_scope"] = np.where(
        neural.method.eq("HIGH"), "reference", "protocol_unified_neural"
    )
    key = ["protocol", "recording", "task", "samples"]
    if set(map(tuple, neural[key].drop_duplicates().to_numpy())) != set(
        map(tuple, shared[key].drop_duplicates().to_numpy())
    ):
        raise ValueError("Shared/neural downstream fixed support views differ")
    result = pd.concat([neural, shared], ignore_index=True)
    if result.duplicated([*key, "method"]).any():
        raise ValueError("Downstream common identity/method repeated")
    return result


def join_shared_temporal(unit, shared, source_reference):
    """Exact physical key join of saved outputs; no concatenation or prediction."""
    result = copy.copy(unit)
    original = unit["frame"].copy()
    high = [axis + "_high" for axis in AXES]
    low = [axis + "_low" for axis in AXES]
    required = [*PHYSICAL_KEYS, *low, *high]
    if set(required) - set(shared) or set([*PHYSICAL_KEYS, *low, *high]) - set(original):
        raise ValueError("Shared temporal source omits exact physical keys/LOW/HIGH")
    if shared.duplicated(PHYSICAL_KEYS).any() or original.duplicated(PHYSICAL_KEYS).any():
        raise ValueError("Shared temporal physical keys are duplicated")
    indexed = shared.set_index(PHYSICAL_KEYS)
    keys = pd.MultiIndex.from_frame(original[PHYSICAL_KEYS])
    if not keys.isin(indexed.index).all():
        raise ValueError("Shared temporal support differs from saved neural rows")
    matched = indexed.reindex(keys).reset_index()
    try:
        pd.testing.assert_frame_equal(
            original[low].reset_index(drop=True), matched[low], check_dtype=False, check_exact=True
        )
        pd.testing.assert_frame_equal(
            original[high].reset_index(drop=True).set_axis(AXES, axis=1),
            matched[high].set_axis(AXES, axis=1),
            check_dtype=False,
            check_exact=True,
        )
        # Preserve every common physical/audit field as well as six-channel values.
        for column in original:
            if column in matched and (
                column.startswith("source_") or column in {"recording_id", "segment_id", "pair_id"}
            ):
                pd.testing.assert_series_equal(
                    original[column].reset_index(drop=True),
                    matched[column],
                    check_names=False,
                    check_dtype=False,
                    check_exact=True,
                )
    except AssertionError as error:
        raise ValueError("Shared temporal physical values/metadata differ") from error
    methods = {
        prefix: meta
        for prefix, meta in unit["methods"].items()
        if meta.get("variant") in {"default", "tuned"}
    }
    for method in METHODS:
        prediction = matched[[method + "_" + axis for axis in AXES]].to_numpy(float)
        if not np.isfinite(prediction).all():
            raise ValueError("Shared temporal outputs are nonfinite")
        if method == "identity":
            if not np.array_equal(prediction, original[low].to_numpy(float)):
                raise ValueError("Shared identity is not the exact common LOW reference")
            continue  # LOW is already drawn once in every panel.
        prefix = "shared_" + method
        for index, axis in enumerate(AXES):
            original[axis + "_" + prefix] = prediction[:, index]
        methods[prefix] = {
            "method": method + " [shared]",
            "family": method,
            "variant": "shared",
            "seed": None,
            "information_scope": SCOPE,
        }
    result.update(
        frame=original,
        methods=methods,
        information_scope=SCOPE,
        source_tables=[*unit.get("source_tables", []), source_reference],
    )
    return result


def attach_shared(tables, reader, roots, shared):
    raw = shared["frames"]
    tables["metrics_shared"] = main_metrics(tables["metrics"], raw["metrics_long"])
    tables["downstream_shared"] = main_downstream(tables["downstream"], raw["downstream_summary"])
    tables["temporal_shared"] = []
    for unit in tables["temporal"]:
        protocol = unit["protocol_id"]
        registry = raw["signal_registry"]
        paths = registry.loc[
            registry.protocol.eq(protocol) & registry.split.eq("test"), "prediction_path"
        ].unique()
        if len(paths) != 1:
            raise ValueError("Shared temporal protocol has ambiguous saved signal source")
        path = reader.path(roots["aux3"] / paths[0])
        frame = pd.read_parquet(path)
        tables["temporal_shared"].append(join_shared_temporal(unit, frame, reader.reference(path)))
    for role in (
        "state_inventory",
        "bank_parameter_totals",
        "latency_summary",
        "latency_raw",
        "workload_latency",
        "alias_audit",
    ):
        tables["shared_" + role] = raw[role]
    tables["_analysis_profile"] = PROFILE
    tables["_shared_controls_index_sha256"] = shared["index_sha256"]
    tables["_shared_coverage"] = {
        **shared["index"]["counts"],
        "neural_split_support_references": len(raw["neural_support_registry"]),
    }
    sources = tables.setdefault("_sources", {})
    for role, key in (
        ("metrics_long", "metrics_shared"),
        ("downstream_summary", "downstream_shared"),
        ("state_inventory", "shared_state_inventory"),
        ("bank_parameter_totals", "shared_bank_parameter_totals"),
        ("latency_summary", "shared_latency_summary"),
        ("latency_raw", "shared_latency_raw"),
        ("workload_latency", "shared_workload_latency"),
        ("alias_audit", "shared_alias_audit"),
    ):
        sources[key] = [reader.reference(roots["aux3"] / shared["descriptors"][role]["path"])]
    sources["metrics_shared"] += sources.get("metrics", [])
    sources["downstream_shared"] += sources.get("downstream", [])
    for key in ("metrics_shared", "downstream_shared"):
        sources[key] += [
            reader.reference(roots["aux3"] / shared["descriptors"][role]["path"])
            for role in ("control_registry", "method_scope")
        ]
    return tables
