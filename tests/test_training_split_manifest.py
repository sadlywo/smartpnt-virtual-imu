from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from virtual_imu.io.training_split_manifest import load_training_split_manifest

SPLITS = ("train", "validation", "test")


def test_loads_three_split_frames_and_preserves_manifest_identity(tmp_path: Path) -> None:
    split_path, strict_path, _, _ = _write_loso_fixture(tmp_path)

    bundle = load_training_split_manifest(split_path, strict_path)

    assert set(bundle.frames) == set(SPLITS)
    assert bundle.frames["train"]["sequence_id"].tolist() == ["af", "af", "as", "as"]
    assert bundle.frames["validation"]["sequence_id"].tolist() == ["bf", "bf"]
    assert bundle.frames["test"]["sequence_id"].tolist() == ["bs", "bs"]
    for split, frame in bundle.frames.items():
        expected = bundle.split_manifest["splits"][split]
        assert set(frame["dataset_id"]) == {record["dataset_id"] for record in expected}
        assert set(frame["pair_id"]) == {record["pair_id"] for record in expected}
    assert bundle.split_manifest_path == split_path.resolve()
    assert bundle.strict_manifest_path == strict_path.resolve()
    assert len(bundle.input_records) == 4


@pytest.mark.parametrize("row_filter", ["all", {"type": "all"}])
def test_accepts_all_row_filter_encodings(tmp_path: Path, row_filter: object) -> None:
    split_path, strict_path, split_manifest, _ = _write_loso_fixture(tmp_path)
    for records in split_manifest["splits"].values():
        for record in records:
            record["row_filter"] = row_filter
    _write_json(split_path, split_manifest)

    bundle = load_training_split_manifest(split_path, strict_path)

    assert len(bundle.frames["train"]) == 4


def test_applies_bounded_time_filters(tmp_path: Path) -> None:
    data_path = tmp_path / "temporal.parquet"
    pd.DataFrame(
        {
            "timestamp_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "training_eligible": [True] * 7,
        }
    ).to_parquet(data_path, index=False)
    identity = {
        "dataset_id": "fusionportable",
        "pair_id": "imu_pair",
        "sequence_id": "mcr_slow",
        "fingerprint": "temporal-fingerprint",
        "training_eligible": True,
    }
    strict_path = tmp_path / "strict.json"
    _write_json(strict_path, {"records": [{**identity, "strict_path": str(data_path)}]})
    split_path = tmp_path / "split.json"
    entries = {
        "train": _temporal_entry(identity, data_path, 0.0, 2.0),
        "validation": _temporal_entry(identity, data_path, 3.0, 4.0),
        "test": _temporal_entry(identity, data_path, 5.0, 6.0),
    }
    entries["validation"]["end_inclusive"] = True
    _write_json(
        split_path,
        {
            "split_kind": "within_sequence_temporal",
            "guard_s": 1.0,
            "splits": {split: [entry] for split, entry in entries.items()},
        },
    )

    bundle = load_training_split_manifest(split_path, strict_path)

    assert bundle.frames["train"]["timestamp_s"].tolist() == [0.0, 1.0]
    assert bundle.frames["validation"]["timestamp_s"].tolist() == [3.0, 4.0]
    assert bundle.frames["test"]["timestamp_s"].tolist() == [5.0]


def test_rejects_missing_parquet_path(tmp_path: Path) -> None:
    split_path, strict_path, split_manifest, strict_manifest = _write_loso_fixture(tmp_path)
    missing_path = tmp_path / "missing.parquet"
    split_manifest["splits"]["test"][0]["path"] = str(missing_path)
    strict_manifest["records"][-1]["strict_path"] = str(missing_path)
    _write_json(split_path, split_manifest)
    _write_json(strict_path, strict_manifest)

    with pytest.raises(FileNotFoundError, match=r"missing\.parquet"):
        load_training_split_manifest(split_path, strict_path)


def test_rejects_strict_manifest_fingerprint_mismatch(tmp_path: Path) -> None:
    split_path, strict_path, split_manifest, _ = _write_loso_fixture(tmp_path)
    split_manifest["splits"]["validation"][0]["fingerprint"] = "wrong"
    _write_json(split_path, split_manifest)

    with pytest.raises(ValueError, match="fingerprint"):
        load_training_split_manifest(split_path, strict_path)


def test_rejects_record_missing_from_strict_manifest(tmp_path: Path) -> None:
    split_path, strict_path, split_manifest, _ = _write_loso_fixture(tmp_path)
    split_manifest["splits"]["validation"][0]["pair_id"] = "unknown_pair"
    _write_json(split_path, split_manifest)

    with pytest.raises(ValueError, match="strict manifest record"):
        load_training_split_manifest(split_path, strict_path)


def test_rejects_ineligible_loaded_row(tmp_path: Path) -> None:
    split_path, strict_path, split_manifest, _ = _write_loso_fixture(tmp_path)
    path = Path(split_manifest["splits"]["train"][0]["path"])
    frame = pd.read_parquet(path)
    frame.loc[1, "training_eligible"] = False
    frame.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="training_eligible"):
        load_training_split_manifest(split_path, strict_path)


def test_rejects_overlapping_group_identities(tmp_path: Path) -> None:
    split_path, strict_path, split_manifest, _ = _write_loso_fixture(tmp_path)
    split_manifest["splits"]["validation"][0]["group_value"] = "af"
    _write_json(split_path, split_manifest)

    with pytest.raises(ValueError, match="overlap"):
        load_training_split_manifest(split_path, strict_path)


def _write_loso_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    strict_records = []
    split_records: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    split_sequences = {"train": ("af", "as"), "validation": ("bf",), "test": ("bs",)}
    for split, sequences in split_sequences.items():
        for sequence in sequences:
            path = tmp_path / f"{sequence}.parquet"
            pd.DataFrame(
                {"timestamp_s": [0.0, 1.0], "value": [1.0, 2.0], "training_eligible": [True, True]}
            ).to_parquet(path, index=False)
            fingerprint = f"fingerprint-{sequence}"
            identity = {
                "dataset_id": "penncosyvio",
                "pair_id": "tango_to_adis",
                "sequence_id": sequence,
                "fingerprint": fingerprint,
                "training_eligible": True,
            }
            strict_records.append({**identity, "strict_path": str(path)})
            split_records[split].append(
                {
                    **identity,
                    "path": str(path),
                    "group_key": "sequence_id",
                    "group_value": sequence,
                    "row_filter": "all",
                }
            )
    strict_manifest = {"records": strict_records}
    split_manifest = {
        "split_kind": "leave_one_sequence_out",
        "group_key": "sequence_id",
        "splits": split_records,
    }
    split_path = tmp_path / "split.json"
    strict_path = tmp_path / "strict.json"
    _write_json(split_path, split_manifest)
    _write_json(strict_path, strict_manifest)
    return split_path, strict_path, split_manifest, strict_manifest


def _temporal_entry(
    identity: dict[str, Any],
    path: Path,
    start_s: float,
    end_s: float,
) -> dict[str, Any]:
    return {
        **identity,
        "path": str(path),
        "time_column": "timestamp_s",
        "time_start_s": start_s,
        "time_end_s": end_s,
        "start_inclusive": True,
        "end_inclusive": False,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
