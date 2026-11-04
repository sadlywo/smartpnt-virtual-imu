"""Derived recording-disjoint protocols over the immutable unified v2 corpus.

Only metadata is written. Original Parquet bytes, canonical sequence identities,
fingerprints, and the base dataset manifest remain unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from virtual_imu.io.multi_imu_splits import validate_no_split_leakage

ROLES = ("train", "validation", "test")
AL_DYNAMIC = ("dyn_01", "dyn_02", "dyn_04", "dyn_05", "dyn_06", "dyn_07", "dyn_08", "dyn_09")
PENN_RECORDINGS = ("af", "as", "bf", "bs")
SMART_RECORDINGS = ("Traj1", "Traj3", "Traj4")
BASE_SHA256 = "5e7f198a57c9370734c4ecce092aa11f797530dded935957324c725c4c752260"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _write_once(path: Path, value: Any) -> None:
    text = _json_text(value)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to replace a different protocol artifact: {path}")
        print(f"SKIPS: identical derived protocol metadata {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"WRITES: derived protocol metadata {path}", flush=True)


def _physical(record: dict) -> str:
    value = record.get("physical_group_id")
    if not isinstance(value, str) or not value:
        raise ValueError("Every record requires an explicit physical_group_id")
    return value


def _recording(record: dict) -> str:
    if record["dataset_id"] == "alimu":
        match = re.search(r"(?:dyn|stat)_\d+", str(record.get("reference_recording_id", "")))
        if not match:
            raise ValueError("ALiMU requires an explicit reference_recording_id")
        return match.group(0)
    return str(record["original_sequence_id"])


def reviewer_protocol_specs(records: list[dict]) -> dict[str, dict]:
    """Return deterministic assignments; all segments/devices move as a block."""
    if not records:
        raise ValueError("The unified catalog must contain records")
    identities = set()
    for record in records:
        identity = (record["dataset_id"], record["pair_id"], record["sequence_id"])
        if identity in identities:
            raise ValueError(f"Duplicate catalog identity: {identity}")
        identities.add(identity)
        _physical(record)
        if record.get("training_eligible") is not True:
            raise ValueError("Reviewer protocols require training-eligible v2 records")
        if record.get("split") not in ROLES:
            raise ValueError("Unknown canonical split role")
        if int(record["rows"]) < 80:
            raise ValueError("The canonical corpus must provide at least 80 samples per segment")
    paired = [r for r in records if r["task"] == "paired_signal"]
    navigation = [r for r in records if r["task"] == "navigation"]
    alimu = [r for r in paired if r["dataset_id"] == "alimu"]
    penn = [r for r in paired if r["dataset_id"] == "penncosyvio"]
    smart = [r for r in navigation if r["dataset_id"] == "smartpnt_msf"]
    for values, expected, label in (
        (alimu, {*AL_DYNAMIC, "stat_11"}, "ALiMU"),
        (penn, set(PENN_RECORDINGS), "Penn"),
        (smart, set(SMART_RECORDINGS), "SmartPNT"),
    ):
        if {_recording(r) for r in values} != expected:
            raise ValueError(f"{label} catalog recording inventory differs from the frozen v2 protocol")
    specs = {}

    def add(identifier: str, task: str, values: list[dict], assignment: dict | None = None, **metadata):
        splits = {role: [] for role in ROLES}
        for record in values:
            role = assignment[_recording(record)] if assignment is not None else record["split"]
            splits[role].append(record)
        if any(not entries for entries in splits.values()):
            raise ValueError(f"Protocol {identifier} has an empty role")
        physical = {role: {_physical(r) for r in values} for role, values in splits.items()}
        if any(physical[left] & physical[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError(f"Protocol {identifier} leaks a physical reference recording")
        recommended_features = ["gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low"]
        specs[identifier] = {"task": task, "splits": splits, "recommended_feature_columns": recommended_features, **metadata}

    for identifier, values in (("pooled", alimu + penn), ("alimu_only", alimu), ("penn_only", penn)):
        add(identifier, "paired", values, protocol_kind="dataset_training_scope")
    for identifier, source, target in (("alimu_to_penn", alimu, penn), ("penn_to_alimu", penn, alimu)):
        values = [r for r in source if r["split"] in {"train", "validation"}] + [r for r in target if r["split"] == "test"]
        add(identifier, "paired", values, protocol_kind="zero_shot_cross_dataset", source_dataset=source[0]["dataset_id"], target_dataset=target[0]["dataset_id"], target_labels_used_for_selection=False)
    for index, test in enumerate(PENN_RECORDINGS):
        validation = PENN_RECORDINGS[(index - 1) % len(PENN_RECORDINGS)]
        assignments = {name: "test" if name == test else "validation" if name == validation else "train" for name in PENN_RECORDINGS}
        add(f"penn_loso_{test}", "paired", penn, assignments, protocol_kind="leave_physical_recording_out", test_recording=test, validation_recording=validation)
    for index, test in enumerate(AL_DYNAMIC):
        validation = AL_DYNAMIC[(index + 1) % len(AL_DYNAMIC)]
        assignments = {name: "test" if name == test else "validation" if name == validation else "train" for name in AL_DYNAMIC}
        assignments["stat_11"] = "train"
        add(f"alimu_loro_{test}", "paired", alimu, assignments, protocol_kind="leave_physical_recording_out", test_recording=test, validation_recording=validation, static_recordings_training_only=["stat_11"])
    for index in range(3):
        test, train, validation = (SMART_RECORDINGS[(index + shift) % 3] for shift in range(3))
        add(f"navigation_rotation_{index}", "navigation", smart, {test: "test", train: "train", validation: "validation"}, protocol_kind="leave_physical_trajectory_out", test_recording=test, validation_recording=validation, independent_trajectory_count=3, folds_are_statistically_independent=False)
    return specs


def _counts(records: list[dict], sizes: list[int], stride: int, rate_hz: float) -> dict:
    intervals = defaultdict(list)
    for record in records:
        intervals[_physical(record)].append((float(record["start_s"]), float(record["end_s"]) + 1 / rate_hz))
    duration = 0.0
    for pieces in intervals.values():
        start, end = sorted(pieces)[0]
        for next_start, next_end in sorted(pieces)[1:]:
            if next_start <= end + 1e-6:
                end = max(end, next_end)
            else:
                duration += end - start
                start, end = next_start, next_end
        duration += end - start
    return {
        "records": len(records), "physical_recordings": len(intervals),
        "rows_across_sensor_records": sum(int(r["rows"]) for r in records),
        "physical_recording_support_s": duration,
        "dataset_ids": sorted({r["dataset_id"] for r in records}),
        "recording_ids": sorted({_recording(r) for r in records}),
        "window_stride": stride,
        "window_counts": {str(size): sum(max(0, (int(r["rows"]) - size) // stride + 1) for r in records) for size in sizes},
        "records_too_short_by_window": {str(size): sum(int(r["rows"]) < size for r in records) for size in sizes},
    }


def generate_reviewer_protocol_splits(config: dict, root: Path) -> dict:
    """Write metadata-only derived protocols and return their absolute paths.

    ``assignment_manifest`` is an opt-in role catalog, not a new data export.
    Training data.dataset_manifest must continue to bind the original manifest;
    a navigation loader must resolve roles from split_manifest explicitly.
    """
    root = root.resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    manifest_path = resolve(config.get("dataset_manifest", "data/processed/unified_imu_v2/manifest.json"))
    strict_path = resolve(config.get("strict_manifest", "data/processed/unified_imu_v2/strict_manifest.json"))
    print(f"READS: canonical v2 dataset manifest {manifest_path}", flush=True)
    print(f"READS: canonical v2 strict manifest {strict_path}", flush=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    strict = json.loads(strict_path.read_text(encoding="utf-8"))
    manifest_hash = _digest(manifest_path)
    if manifest_hash != config.get("expected_dataset_sha256", BASE_SHA256):
        raise ValueError("Base dataset manifest checksum mismatch; no derived protocols written")
    if manifest.get("dataset_id") != "unified_imu_v2" or float(manifest.get("rate_hz", 0)) != 40.0:
        raise ValueError("Only the immutable native-40Hz unified_imu_v2 corpus is supported")
    source_index = {(r["dataset_id"], r["pair_id"], r["sequence_id"]): r for r in strict["records"]}
    records = manifest["records"]
    for record in records:
        identity = (record["dataset_id"], record["pair_id"], record["sequence_id"])
        matched = source_index.get(identity)
        if record["task"] == "navigation":
            # The canonical strict catalog intentionally contains only paired
            # labels. Navigation eligibility is bound by the original catalog.
            if record["fingerprint"] != record["sha256"] or record["path"] != record["strict_path"]:
                raise ValueError(f"Canonical navigation record metadata disagree: {identity}")
            continue
        if matched is None or record["fingerprint"] != matched["fingerprint"] or record["path"] != matched["strict_path"]:
            raise ValueError(f"Base strict manifest disagrees with canonical record {identity}")
    specs = reviewer_protocol_specs(records)
    stride = int(config.get("stride", 40))
    sizes = list(config.get("window_sizes", [80, 160, 320]))
    if stride <= 0 or not sizes or any(type(size) is not int or size < 80 for size in sizes):
        raise ValueError("Require positive stride and integer window sizes >=80")
    output = resolve(config.get("output_root", "runs/reviewer_protocol_20261024/splits"))
    if not output.is_relative_to(root / "runs"):
        raise ValueError("Derived protocol artifacts must remain under the project's runs directory")
    canonical = {
        "dataset_manifest": str(manifest_path), "dataset_manifest_sha256": manifest_hash,
        "strict_manifest": str(strict_path), "strict_manifest_sha256": _digest(strict_path),
    }
    result = {"schema_version": 1, "seed": int(config.get("seed", 20261024)), "base_dataset_manifest": str(manifest_path), "base_dataset_sha256": manifest_hash, "protocols": {}}
    for identifier, spec in specs.items():
        protocol_root = output / identifier
        derived_records, entries = [], {role: [] for role in ROLES}
        for role, values in spec["splits"].items():
            for record in values:
                derived = copy.deepcopy(record)
                derived["canonical_split"] = record["split"]
                derived["split"] = role
                derived_records.append(derived)
                entries[role].append({**derived, "group_key": "physical_group_id", "group_value": _physical(record), "row_filter": "all"})
        metadata = {key: value for key, value in spec.items() if key != "splits"}
        split = {"schema_version": 1, "split_kind": "leave_one_sequence_out", "fold_id": identifier, "group_key": "physical_group_id", "eligibility_required": True, "seed": result["seed"], "canonical_provenance": canonical, "derived_assignment": True, "metadata": metadata, "splits": entries}
        validate_no_split_leakage(split)
        split_path, derived_strict_path, assignment_path = (protocol_root / name for name in ("split_manifest.json", "strict_manifest.json", "assignment_manifest.json"))
        _write_once(split_path, split)
        _write_once(derived_strict_path, {"records": derived_records, "build_run": manifest["build_run"], "canonical_provenance": canonical, "derived_assignment": True})
        _write_once(assignment_path, {"schema_version": 1, "schema_kind": "reviewer_assignment_catalog", "dataset_id": manifest["dataset_id"], "rate_hz": manifest["rate_hz"], "canonical_provenance": canonical, "derived_assignment": True, "records": derived_records})
        result["protocols"][identifier] = {
            **metadata, "split_manifest": str(split_path), "strict_manifest": str(derived_strict_path),
            "assignment_manifest": str(assignment_path), "counts": {role: _counts(values, sizes, stride, 40.0) for role, values in spec["splits"].items()},
            "roles": {role: sorted({_recording(r) for r in values}) for role, values in spec["splits"].items()},
        }
    result["index_path"] = str(output / "protocol_index.json")
    _write_once(Path(result["index_path"]), result)
    _write_once(output / "config_snapshot.json", config)
    if _digest(manifest_path) != manifest_hash:
        raise RuntimeError("Canonical manifest changed while generating protocols")
    print("SKIPS: original Parquet rewrites, canonical manifest changes, target fitting, old diagnostic LOSO data", flush=True)
    return result
