from __future__ import annotations

import json
from pathlib import Path

import pytest

from virtual_imu.io.multi_imu_splits import (
    fusion_temporal_manifest,
    penn_loso_manifests,
    validate_no_split_leakage,
)


def test_penn_loso_writes_four_disjoint_folds_with_required_default(tmp_path: Path) -> None:
    paths = penn_loso_manifests(_strict_manifest(), tmp_path)

    assert len(paths) == 4
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for manifest in manifests:
        validate_no_split_leakage(manifest)
        groups = {
            split: {entry["group_value"] for entry in manifest["splits"][split]}
            for split in ("train", "validation", "test")
        }
        assert groups["train"].isdisjoint(groups["validation"] | groups["test"])
        assert groups["validation"].isdisjoint(groups["test"])
    default = next(manifest for manifest in manifests if manifest["fold_id"] == "default_bs")
    assert {entry["group_value"] for entry in default["splits"]["train"]} == {"af", "as"}
    assert {entry["group_value"] for entry in default["splits"]["validation"]} == {"bf"}
    assert {entry["group_value"] for entry in default["splits"]["test"]} == {"bs"}


def test_fusion_temporal_boundaries_are_shared_guarded_and_static_train_only(
    tmp_path: Path,
) -> None:
    path = fusion_temporal_manifest(_strict_manifest(), tmp_path, guard_s=10.0)
    manifest = json.loads(path.read_text(encoding="utf-8"))

    validate_no_split_leakage(manifest)
    dynamic_by_split = {
        split: [entry for entry in manifest["splits"][split] if entry["sequence_id"] == "mcr_slow"]
        for split in ("train", "validation", "test")
    }
    assert all(len(entries) == 3 for entries in dynamic_by_split.values())
    for split, entries in dynamic_by_split.items():
        boundaries = {(entry["time_start_s"], entry["time_end_s"]) for entry in entries}
        assert len(boundaries) == 1, split
    train_bounds = (
        dynamic_by_split["train"][0]["time_start_s"],
        dynamic_by_split["train"][0]["time_end_s"],
    )
    val_bounds = (
        dynamic_by_split["validation"][0]["time_start_s"],
        dynamic_by_split["validation"][0]["time_end_s"],
    )
    test_bounds = (
        dynamic_by_split["test"][0]["time_start_s"],
        dynamic_by_split["test"][0]["time_end_s"],
    )
    assert val_bounds[0] - train_bounds[1] == pytest.approx(10.0)
    assert test_bounds[0] - val_bounds[1] == pytest.approx(10.0)
    retained = [train_bounds[1] - train_bounds[0], val_bounds[1] - val_bounds[0], test_bounds[1] - test_bounds[0]]
    assert retained == pytest.approx([48.0, 16.0, 16.0])
    assert any(entry["sequence_id"] == "static" for entry in manifest["splits"]["train"])
    assert all(entry["sequence_id"] != "static" for entry in manifest["splits"]["validation"])
    assert all(entry["sequence_id"] != "static" for entry in manifest["splits"]["test"])


def test_validate_no_split_leakage_rejects_group_overlap() -> None:
    manifest = {
        "split_kind": "leave_one_sequence_out",
        "splits": {
            "train": [{"path": "a.parquet", "group_value": "af", "training_eligible": True}],
            "validation": [{"path": "b.parquet", "group_value": "af", "training_eligible": True}],
            "test": [],
        },
    }

    with pytest.raises(ValueError, match="overlap"):
        validate_no_split_leakage(manifest)


def test_validate_no_split_leakage_rejects_temporal_guard_violation() -> None:
    manifest = {
        "split_kind": "within_sequence_temporal",
        "guard_s": 10.0,
        "splits": {
            "train": [_temporal_entry("p", 0.0, 60.0)],
            "validation": [_temporal_entry("p", 65.0, 80.0)],
            "test": [_temporal_entry("p", 90.0, 100.0)],
        },
    }

    with pytest.raises(ValueError, match="guard"):
        validate_no_split_leakage(manifest)


def _strict_manifest() -> dict[str, object]:
    records = []
    for sequence in ("af", "as", "bf", "bs"):
        records.append(
            {
                "dataset_id": "penncosyvio",
                "pair_id": "tango_to_adis",
                "sequence_id": sequence,
                "split_group": sequence,
                "strict_path": f"penn/{sequence}.parquet",
                "training_eligible": True,
                "start_s": 0.0,
                "end_s": 100.0,
            }
        )
    for pair_id in ("ouster", "davis_left", "davis_right"):
        records.extend(
            [
                {
                    "dataset_id": "fusionportable",
                    "pair_id": pair_id,
                    "sequence_id": "mcr_slow",
                    "split_group": "mcr_slow",
                    "source_kind": "dynamic",
                    "strict_path": f"fusion/{pair_id}/mcr.parquet",
                    "training_eligible": True,
                    "start_s": 0.0,
                    "end_s": 100.0,
                },
                {
                    "dataset_id": "fusionportable",
                    "pair_id": pair_id,
                    "sequence_id": "static",
                    "split_group": "static",
                    "source_kind": "static",
                    "strict_path": f"fusion/{pair_id}/static.parquet",
                    "training_eligible": True,
                    "start_s": 0.0,
                    "end_s": 1000.0,
                },
            ]
        )
    return {"records": records}


def _temporal_entry(pair_id: str, start_s: float, end_s: float) -> dict[str, object]:
    return {
        "path": f"{pair_id}.parquet",
        "pair_id": pair_id,
        "sequence_id": "mcr",
        "time_start_s": start_s,
        "time_end_s": end_s,
        "training_eligible": True,
    }
