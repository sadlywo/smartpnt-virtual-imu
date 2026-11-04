from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from virtual_imu.training.dataset import LOW_IMU_COLUMNS, TARGET_COLUMNS
from virtual_imu.training.trainer import train_model


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(42)
    strict_records = []
    split_records = {"train": [], "validation": [], "test": []}
    assignments = {"train": ("af", "as"), "validation": ("bf",), "test": ("bs",)}
    for split_name, sequences in assignments.items():
        for sequence in sequences:
            low = rng.normal(size=(24, 6)).astype(np.float32)
            high = low + np.float32(0.05)
            frame = pd.DataFrame(low, columns=LOW_IMU_COLUMNS)
            for index, column in enumerate(TARGET_COLUMNS):
                frame[column] = high[:, index]
            frame["timestamp_s"] = np.arange(24) * 0.01
            frame["sequence_id"] = sequence
            frame["low_sensor"] = "Tango Top IMU"
            frame["training_eligible"] = True
            path = tmp_path / f"{sequence}.parquet"
            frame.to_parquet(path, index=False)
            fingerprint = f"fingerprint-{sequence}"
            identity = {
                "dataset_id": "penncosyvio",
                "pair_id": "tango_top_to_adis16488",
                "sequence_id": sequence,
                "fingerprint": fingerprint,
                "training_eligible": True,
            }
            strict_records.append({**identity, "strict_path": str(path)})
            split_records[split_name].append(
                {
                    **identity,
                    "path": str(path),
                    "group_key": "sequence_id",
                    "group_value": sequence,
                    "row_filter": "all",
                }
            )
    strict_path = tmp_path / "strict.json"
    split_path = tmp_path / "split.json"
    strict_path.write_text(json.dumps({"records": strict_records}), encoding="utf-8")
    split_path.write_text(
        json.dumps(
            {
                "split_kind": "leave_one_sequence_out",
                "group_key": "sequence_id",
                "splits": split_records,
            }
        ),
        encoding="utf-8",
    )
    return split_path, strict_path


def test_manifest_native_one_epoch_training_snapshots_provenance_and_context(tmp_path: Path):
    split_path, strict_path = _write_fixture(tmp_path)
    config = {
        "seed": 20260712,
        "experiment_name": "manifest_smoke",
        "data": {
            "split_manifest": str(split_path),
            "strict_manifest": str(strict_path),
            "feature_columns": LOW_IMU_COLUMNS,
            "window_size": 8,
            "stride": 8,
            "batch_size": 4,
            "num_workers": 0,
            "static_context": {
                "enabled": True,
                "columns": LOW_IMU_COLUMNS,
                "window_size": 8,
                "selection": "auto_low_motion",
                "reference_scope": "train_sensor",
            },
        },
        "model": {
            "type": "bias_drift_tcn",
            "target_mode": "residual",
            "hidden_channels": 4,
            "levels": 1,
            "kernel_size": 3,
            "dropout": 0.0,
            "static_channels": 6,
            "bias_branch_mode": "offset",
        },
        "training": {
            "epochs": 1,
            "lr": 0.001,
            "loss": "huber",
            "checkpoint_metric": "val_loss",
            "device": "cpu",
            "early_stopping_patience": 2,
        },
        "output": {"runs_dir": str(tmp_path / "runs")},
    }

    run_dir = train_model(config)
    checkpoint = torch.load(run_dir / "best.ckpt", map_location="cpu")
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))

    assert checkpoint["feature_columns"] == LOW_IMU_COLUMNS
    assert checkpoint["split_provenance"] == {
        "train": ["af", "as"],
        "validation": ["bf"],
        "test": ["bs"],
    }
    assert set(checkpoint["static_windows"]) == {"Tango Top IMU"}
    assert checkpoint["static_window_metadata"]["Tango Top IMU"]["sequence_id"] in {"af", "as"}
    assert provenance["split_provenance"] == checkpoint["split_provenance"]
    assert (run_dir / "split_manifest.json").exists()
    assert (run_dir / "strict_manifest.json").exists()
