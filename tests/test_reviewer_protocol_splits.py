"""Recording-level protection for metadata-only reviewer protocols."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from virtual_imu.io.reviewer_protocol_splits import (
    AL_DYNAMIC,
    generate_reviewer_protocol_splits,
    reviewer_protocol_specs,
)


def catalog() -> list[dict]:
    records = []
    for dataset, recordings, sensors in (
        ("alimu", [*AL_DYNAMIC, "stat_11"], ["BMX160", "MPU9250", "LSM9DS1"]),
        ("penncosyvio", ["af", "as", "bf", "bs"], ["Tango"]),
        ("smartpnt_msf", ["Traj1", "Traj3", "Traj4"], ["Smart"]),
    ):
        for index, recording in enumerate(recordings):
            if dataset == "alimu":
                role = "validation" if recording == "dyn_07" else "test" if recording in {"dyn_08", "dyn_09"} else "train"
            elif dataset == "penncosyvio":
                role = "validation" if recording == "bf" else "test" if recording == "bs" else "train"
            else:
                role = {"Traj1": "test", "Traj3": "train", "Traj4": "validation"}[recording]
            for sensor in sensors:
                # Two navigation segments verify that no within-trajectory split is possible.
                for segment in range(2 if dataset == "smartpnt_msf" else 1):
                    sequence = f"{dataset}_{recording}_{sensor}_{segment}"
                    path = f"data/processed/unified_imu_v2/{sequence}.parquet"
                    records.append({
                        "dataset_id": dataset, "pair_id": sensor, "sequence_id": sequence,
                        "original_sequence_id": recording, "reference_recording_id": recording,
                        "task": "navigation" if dataset == "smartpnt_msf" else "paired_signal",
                        "physical_group_id": f"{dataset}/{recording}", "training_eligible": True,
                        "split": role, "rows": 200, "start_s": segment * 10.0,
                        "end_s": segment * 10.0 + 4.975, "path": path, "strict_path": path,
                        "fingerprint": f"hash_{index}_{sensor}_{segment}", "sha256": f"hash_{index}_{sensor}_{segment}",
                    })
    return records


def test_all_protocols_keep_shared_reference_and_segments_together():
    protocols = reviewer_protocol_specs(catalog())
    assert len(protocols) == 20
    for protocol in protocols.values():
        seen = {}
        for role, records in protocol["splits"].items():
            for record in records:
                group = record["physical_group_id"]
                assert group not in seen or seen[group] == role
                seen[group] = role
    penn = protocols["penn_loso_bs"]["splits"]
    assert {r["original_sequence_id"] for r in penn["validation"]} == {"bf"}
    assert {r["original_sequence_id"] for r in penn["train"]} == {"af", "as"}
    for name, protocol in protocols.items():
        if name.startswith("alimu_loro"):
            assert sum(r["reference_recording_id"] == "stat_11" for r in protocol["splits"]["train"]) == 3
            assert len(protocol["splits"]["test"]) == 3


def test_zero_shot_never_uses_target_train_or_validation_recordings():
    specs = reviewer_protocol_specs(catalog())
    for identifier, source, target in (("alimu_to_penn", "alimu", "penncosyvio"), ("penn_to_alimu", "penncosyvio", "alimu")):
        splits = specs[identifier]["splits"]
        assert {r["dataset_id"] for r in splits["train"] + splits["validation"]} == {source}
        assert {r["dataset_id"] for r in splits["test"]} == {target}
        assert all(r["split"] == "test" for r in splits["test"])


def test_malformed_or_shared_canonical_physical_groups_fail():
    records = catalog()
    records[0].pop("physical_group_id")
    with pytest.raises(ValueError, match="physical_group_id"):
        reviewer_protocol_specs(records)
    records = catalog()
    records[0]["physical_group_id"] = "alimu/dyn_08"
    with pytest.raises(ValueError, match="leaks a physical"):
        reviewer_protocol_specs(records)


def test_generation_preserves_inputs_reassigns_metadata_and_is_repeatable(tmp_path: Path):
    inputs = tmp_path / "data"
    inputs.mkdir()
    records = catalog()
    manifest = inputs / "manifest.json"
    strict = inputs / "strict.json"
    manifest.write_text(json.dumps({"dataset_id": "unified_imu_v2", "rate_hz": 40, "build_run": "runs/base", "records": records}), encoding="utf-8")
    strict.write_text(json.dumps({"records": [r for r in records if r["task"] == "paired_signal"]}), encoding="utf-8")
    original = manifest.read_bytes()
    checksum = hashlib.sha256(original).hexdigest()
    config = {"dataset_manifest": "data/manifest.json", "strict_manifest": "data/strict.json", "expected_dataset_sha256": checksum}
    result = generate_reviewer_protocol_splits(config, tmp_path)
    assert manifest.read_bytes() == original
    assert generate_reviewer_protocol_splits(config, tmp_path) == result
    pooled_counts = result["protocols"]["pooled"]["counts"]["test"]
    assert pooled_counts["physical_recordings"] == 3
    assert pooled_counts["physical_recording_support_s"] == pytest.approx(15.0)
    assert pooled_counts["window_counts"]["320"] == 0
    derived = json.loads(Path(result["protocols"]["penn_loso_bs"]["assignment_manifest"]).read_text())
    af = next(r for r in derived["records"] if r["original_sequence_id"] == "af")
    assert af["canonical_split"] == "train" and af["split"] == "train"
    as_rotation = json.loads(Path(result["protocols"]["penn_loso_as"]["assignment_manifest"]).read_text())
    af_validation = next(r for r in as_rotation["records"] if r["original_sequence_id"] == "af")
    assert af_validation["canonical_split"] == "train" and af_validation["split"] == "validation"
    assert af["path"] in {r["path"] for r in records}
    config["expected_dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        generate_reviewer_protocol_splits(config, tmp_path)
