from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from virtual_imu.config import load_config
from virtual_imu.training.penn_loso_runner import (
    PENN_METHODS,
    apply_smoke_overrides,
    run_penn_loso_comparison,
    validate_penn_loso_config,
)


def test_default_comparison_config_is_valid() -> None:
    config = load_config("configs/penn_loso_comparison_default.yaml")

    validate_penn_loso_config(config)

    invalid = deepcopy(config)
    del invalid["models"]["bias_drift_tcn"]
    with pytest.raises(ValueError, match="models"):
        validate_penn_loso_config(invalid)


def test_smoke_overrides_one_epoch_and_small_network_without_mutation() -> None:
    original = _model_config("tcn", Path("split.json"), Path("strict.json"))
    untouched = deepcopy(original)

    smoke = apply_smoke_overrides(original)

    assert original == untouched
    assert smoke["training"]["epochs"] == 1
    assert smoke["training"]["device"] == "cpu"
    assert smoke["model"]["hidden_channels"] <= 8
    assert smoke["model"]["levels"] == 1
    assert smoke["data"]["window_size"] <= 16


def test_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_penn_loso_comparison.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--smoke" in result.stdout


def test_smoke_run_writes_six_method_comparison_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_path, strict_path = _write_strict_fixture(tmp_path)
    tcn_path = tmp_path / "tcn.yaml"
    bias_path = tmp_path / "bias.yaml"
    _write_yaml(tcn_path, _model_config("tcn", split_path, strict_path))
    _write_yaml(bias_path, _model_config("bias_drift_tcn", split_path, strict_path))
    runs_dir = tmp_path / "runs"
    latest_dir = tmp_path / "outputs" / "latest"
    latest_run_file = tmp_path / "outputs" / "latest_run.txt"
    config_path = tmp_path / "comparison.yaml"
    config = {
        "seed": 20260712,
        "experiment_name": "penn_test_comparison",
        "split_manifest": str(split_path),
        "strict_manifest": str(strict_path),
        "models": {"tcn": str(tcn_path), "bias_drift_tcn": str(bias_path)},
        "baselines": {"fir_taps": 3, "ridge_alpha": 1e-6},
        "output": {
            "runs_dir": str(runs_dir),
            "latest_dir": str(latest_dir),
            "latest_run_file": str(latest_run_file),
        },
    }
    _write_yaml(config_path, config)
    monkeypatch.setattr("virtual_imu.training.trainer._plot_loss_curve", lambda *_: None)

    run_dir = run_penn_loso_comparison(config, config_path=config_path, smoke=True)

    comparison = pd.read_csv(run_dir / "comparison.csv")
    assert set(comparison["method"]) == set(PENN_METHODS)
    assert len(comparison) == 6
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["split_provenance"] == {
        "train": ["af", "as"],
        "validation": ["bf"],
        "test": ["bs"],
    }
    assert provenance["baseline_fit"]["sequences"] == ["af", "as"]
    assert provenance["smoke"] is True
    static_sources = provenance["models"]["bias_drift_tcn"]["static_context_sources"]
    assert {source["sequence_id"] for source in static_sources.values()} <= {"af", "as"}
    assert set(provenance["artifacts"]["predictions"]) == set(PENN_METHODS)
    assert set(provenance["artifacts"]["metrics"]) == set(PENN_METHODS)
    for method in ("tcn", "bias_drift_tcn"):
        checkpoint_path = Path(provenance["models"][method]["best_checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert checkpoint["best_epoch"] == 1
        assert checkpoint["config"]["training"]["epochs"] == 1
        assert checkpoint["split_provenance"]["test"] == ["bs"]
    assert latest_run_file.read_text(encoding="utf-8").strip() == str(run_dir.resolve())
    assert {path.name for path in latest_dir.iterdir()} == {
        "comparison.csv",
        "comparison.json",
        "comparison.md",
        "provenance.json",
    }


def _model_config(model_type: str, split_path: Path, strict_path: Path) -> dict[str, Any]:
    static_context = {
        "enabled": True,
        "columns": ["gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low"],
        "window_size": 8,
        "selection": "auto_low_motion",
        "reference_scope": "train_sensor",
    }
    model = {
        "type": model_type,
        "target_mode": "residual",
        "hidden_channels": 12,
        "levels": 2,
        "kernel_size": 3,
        "dropout": 0.0,
    }
    data: dict[str, Any] = {
        "split_manifest": str(split_path),
        "strict_manifest": str(strict_path),
        "feature_columns": [
            "gx_low",
            "gy_low",
            "gz_low",
            "ax_low",
            "ay_low",
            "az_low",
        ],
        "window_size": 8,
        "stride": 8,
        "batch_size": 4,
        "num_workers": 0,
    }
    if model_type == "bias_drift_tcn":
        data["static_context"] = static_context
        model.update(
            {
                "static_channels": 6,
                "use_bias_branch": True,
                "use_drift_branch": True,
                "bias_branch_mode": "offset",
            }
        )
    return {
        "seed": 20260712,
        "experiment_name": f"test_{model_type}",
        "data": data,
        "model": model,
        "training": {
            "epochs": 3,
            "lr": 1e-3,
            "loss": "huber",
            "checkpoint_metric": "val_loss",
            "early_stopping_patience": 2,
            "device": "cpu",
        },
        "output": {"runs_dir": "unused"},
    }


def _write_strict_fixture(tmp_path: Path) -> tuple[Path, Path]:
    strict_records = []
    split_records: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    split_sequences = {"train": ("af", "as"), "validation": ("bf",), "test": ("bs",)}
    for split, sequences in split_sequences.items():
        for sequence_index, sequence in enumerate(sequences):
            path = tmp_path / f"{sequence}.parquet"
            _paired_frame(sequence, sequence_index).to_parquet(path, index=False)
            identity = {
                "dataset_id": "penncosyvio",
                "pair_id": "tango_to_adis",
                "sequence_id": sequence,
                "fingerprint": f"fingerprint-{sequence}",
                "training_eligible": True,
            }
            strict_records.append({**identity, "strict_path": str(path)})
            split_records[split].append(
                {
                    **identity,
                    "path": str(path),
                    "group_key": "sequence_id",
                    "group_value": sequence,
                    "row_filter": {"type": "all"},
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


def _paired_frame(sequence_id: str, offset: int) -> pd.DataFrame:
    count = 32
    timestamp = np.arange(count, dtype=float) * 0.01
    phase = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False) + offset * 0.2
    low = np.column_stack(
        [
            np.sin(phase),
            np.cos(phase),
            np.sin(phase * 0.5),
            0.2 * np.cos(phase * 0.5),
            0.3 * np.sin(phase * 1.5),
            9.81 + 0.1 * np.cos(phase),
        ]
    )
    high = low * np.asarray([1.02, 0.98, 1.01, 1.03, 0.97, 1.001]) + np.asarray(
        [0.01, -0.02, 0.005, 0.02, -0.01, 0.03]
    )
    columns = ["gx", "gy", "gz", "ax", "ay", "az"]
    frame = pd.DataFrame({"timestamp_s": timestamp})
    for index, axis in enumerate(columns):
        frame[f"{axis}_low"] = low[:, index]
        frame[f"{axis}_high"] = high[:, index]
    frame["sequence_id"] = sequence_id
    frame["low_sensor"] = "tango"
    frame["training_eligible"] = True
    return frame


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
