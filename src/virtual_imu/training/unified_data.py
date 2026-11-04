"""Sampling support for differently sized measured IMU datasets."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from torch.utils.data import WeightedRandomSampler

from virtual_imu.io.multi_imu_splits import validate_no_split_leakage
from virtual_imu.io.unified_dataset import file_sha256, verify_unified_dataset
from virtual_imu.utils.paths import project_root_from_file


def dataset_balanced_sampler(
    frame: pd.DataFrame,
    *,
    window_size: int,
    stride: int,
    seed: int,
) -> WeightedRandomSampler:
    """Assign equal expected sampling mass per dataset over actual windows."""
    identities = []
    for _, group in frame.groupby("sequence_id", sort=False):
        if group.dataset_id.nunique() != 1:
            raise ValueError("A training segment must belong to one dataset")
        count = max(0, (len(group) - window_size) // stride + 1)
        identities.extend([str(group.dataset_id.iloc[0])] * count)
    if not identities:
        raise ValueError("No windows available for dataset-balanced sampling")
    counts = Counter(identities)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        [1 / counts[value] for value in identities],
        num_samples=len(identities),
        replacement=True,
        generator=generator,
    )


def unified_navigation_frame(data_config: dict) -> pd.DataFrame:
    """Read only hash-verified navigation records declared in the task manifest."""
    root = project_root_from_file(__file__)
    manifest_path = root / data_config["dataset_manifest"]
    if data_config.get("load_splits"):
        verify_unified_dataset(manifest_path, root, validate_payloads=False)
        logger.info("SKIPS full-corpus label decoding during selected-role loading; immutable file hashes still checked")
    else:
        verify_unified_dataset(manifest_path, root)
    logger.info("READS navigation split manifest {}", data_config["split_manifest"])
    split = json.loads((root / data_config["split_manifest"]).read_text(encoding="utf-8"))
    validate_no_split_leakage(split)
    catalog = json.loads(manifest_path.read_text(encoding="utf-8"))
    record_index = {r["sequence_id"]: r for r in catalog["records"] if r["task"] == "navigation"}
    derived = bool(data_config.get("assignment_manifest"))
    if derived:
        assignment_path = root / data_config["assignment_manifest"]
        logger.info("READS derived navigation assignment {}", assignment_path)
        assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
        provenance = assignment["canonical_provenance"]
        canonical = root / provenance["dataset_manifest"]
        if canonical.resolve() != manifest_path.resolve() or file_sha256(canonical) != provenance["dataset_manifest_sha256"]:
            raise ValueError("Derived navigation assignments do not bind the original dataset")
        assigned = {}
        for record in assignment["records"]:
            expected = record_index.get(record["sequence_id"])
            if expected is None or any(record.get(key) != expected.get(key) for key in ("path", "sha256", "dataset_id", "physical_group_id", "original_sequence_id")):
                raise ValueError("Derived navigation assignments changed immutable record identity")
            if record["sequence_id"] in assigned:
                raise ValueError("Duplicate derived navigation assignment")
            assigned[record["sequence_id"]] = record
        record_index = assigned
    selected = tuple(data_config.get("load_splits", ("train", "validation", "test")))
    if not selected or len(set(selected)) != len(selected) or set(selected) - {"train", "validation", "test"}:
        raise ValueError("Invalid selected navigation load_splits")
    frames = []
    for name, config_name in (
        ("train", "train_sequences"),
        ("validation", "val_sequences"),
        ("test", "test_sequences"),
    ):
        entries = split["splits"][name]
        requested = set(data_config[config_name])
        if requested != {r["original_sequence_id"] for r in entries}:
            raise ValueError(f"Navigation config sequences differ from declared {name} split")
        if name not in selected:
            logger.info("SKIPS navigation {} payloads during validation-only training", name)
            continue
        for entry in entries:
            expected = record_index.get(entry["sequence_id"])
            if (
                expected is None
                or expected["path"] != entry["path"]
                or expected["sha256"] != entry["sha256"]
                or expected["split"] != name
            ):
                raise ValueError("Navigation split does not match the dataset catalog")
            logger.info("READS declared navigation record {}", entry["path"])
            frame = pd.read_parquet(root / entry["path"])
            if derived:
                if "split" in frame:
                    frame["canonical_split"] = frame["split"]
                frame["split"] = name
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def snapshot_unified_training_inputs(data_config: dict, run_dir: Path) -> dict:
    """Copy the exact corpus/split metadata into the training run for provenance."""
    if "dataset_manifest" not in data_config:
        return {}
    root = project_root_from_file(__file__)
    verification = (
        verify_unified_dataset(root / data_config["dataset_manifest"], root, validate_payloads=False)
        if data_config.get("load_splits")
        else verify_unified_dataset(root / data_config["dataset_manifest"], root)
    )
    logger.info("WRITES training data verification {}", run_dir / "data_verification.json")
    (run_dir / "data_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    references = {}
    catalog = {}
    for key in ("dataset_manifest", "split_manifest", "strict_manifest", "assignment_manifest"):
        if not data_config.get(key):
            continue
        path = root / data_config[key]
        logger.info("READS training provenance {}", path)
        if key == "dataset_manifest":
            catalog = json.loads(path.read_text(encoding="utf-8"))
        destination = run_dir / (
            "data_manifest.json" if key == "dataset_manifest" else f"{key}.json"
        )
        logger.info("WRITES training metadata snapshot {}", destination)
        shutil.copy2(path, destination)
        references[key] = {"path": data_config[key], "sha256": file_sha256(path)}
    provenance = {
        "unified_dataset": {
            "dataset_id": catalog["dataset_id"],
            "build_run": catalog["build_run"],
            "source_state_sha256": catalog.get("source_state_sha256"),
            "inputs": references,
        }
    }
    launch_run = os.environ.get("VIRTUAL_IMU_SERVER_LAUNCH_RUN")
    if launch_run:
        provenance["server_launch_run"] = launch_run
        launch_path = (root / launch_run).resolve()
        if not launch_path.is_relative_to(root / "runs"):
            raise ValueError("Server launch provenance must be under runs")
        if (launch_path / "source_bundle.json").is_file():
            bundle = launch_path / "source_bundle.json"
            shutil.copy2(bundle, run_dir / "source_bundle.json")
            provenance["server_source_bundle"] = {
                "path": str(bundle.relative_to(root)),
                "sha256": file_sha256(bundle),
            }
    return provenance
