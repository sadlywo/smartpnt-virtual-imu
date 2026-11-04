from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from virtual_imu.config import load_config, save_config
from virtual_imu.eval.baselines import (
    fit_system_id_baseline_state,
    predict_system_id_from_state,
)
from virtual_imu.io.training_split_manifest import TrainingSplitBundle
from virtual_imu.training import penn_matrix_config
from virtual_imu.training.penn_matrix_config import FOLD_ASSIGNMENTS, PENN_NEURAL_METHODS
from virtual_imu.training.penn_matrix_runner import run_penn_matrix

LOW = ["gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low"]
HIGH = ["gx_high", "gy_high", "gz_high", "ax_high", "ay_high", "az_high"]
VIRTUAL = [
    "gx_virtual",
    "gy_virtual",
    "gz_virtual",
    "ax_virtual",
    "ay_virtual",
    "az_virtual",
]


def _frame(sequence_id: str, offset: float) -> pd.DataFrame:
    time = np.arange(32, dtype=float) * 0.01
    values = np.column_stack(
        [np.sin(time * (axis + 1) * 3.0) + offset + axis for axis in range(6)]
    )
    frame = pd.DataFrame(values, columns=LOW)
    for axis, column in enumerate(HIGH):
        frame[column] = values[:, axis] * (1.0 + 0.01 * axis) - 0.02
    frame.insert(0, "timestamp_s", time)
    frame["sequence_id"] = sequence_id
    frame["low_sensor"] = "low-a"
    return frame


def _base_config() -> dict[str, Any]:
    return {
        "seed": 1,
        "experiment_name": "base",
        "data": {
            "feature_columns": LOW,
            "window_size": 128,
            "stride": 64,
            "batch_size": 8,
            "num_workers": 0,
            "static_context": {
                "enabled": True,
                "columns": LOW,
                "window_size": 2048,
                "selection": "auto_low_motion",
                "reference_scope": "train_sensor",
            },
        },
        "model": {
            "type": "bias_drift_tcn",
            "target_mode": "residual",
            "hidden_channels": 48,
            "levels": 4,
            "kernel_size": 5,
            "dropout": 0.05,
        },
        "training": {
            "epochs": 30,
            "device": "auto",
            "bias_auxiliary": {
                "enabled": False,
                "weight": 0.0,
                "stationary_fraction_threshold": 0.8,
            },
        },
        "output": {"runs_dir": "runs"},
    }


@pytest.fixture
def matrix_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    frames = {name: _frame(name, index * 0.1) for index, name in enumerate(("af", "as", "bf", "bs"))}
    strict_path = tmp_path / "strict.json"
    strict_path.write_text('{"records": []}', encoding="utf-8")
    base_path = tmp_path / "base.yaml"
    save_config(_base_config(), base_path)

    folds: dict[str, Any] = {}
    bundles: dict[str, TrainingSplitBundle] = {}
    for fold_id, assignment in FOLD_ASSIGNMENTS.items():
        manifest_path = tmp_path / f"{fold_id}.json"
        manifest_path.write_text(json.dumps({"fold_id": fold_id}), encoding="utf-8")
        split_manifest = {
            "fold_id": fold_id,
            "splits": {
                split: [{"sequence_id": sequence} for sequence in sequences]
                for split, sequences in assignment.items()
            },
        }
        split_frames = {
            split: pd.concat([frames[sequence] for sequence in sequences], ignore_index=True)
            for split, sequences in assignment.items()
        }
        bundles[str(manifest_path.resolve())] = TrainingSplitBundle(
            frames=split_frames,
            split_manifest=split_manifest,
            strict_manifest={"records": []},
            split_manifest_path=manifest_path,
            strict_manifest_path=strict_path,
            input_records=(),
        )
        folds[fold_id] = {
            "manifest": str(manifest_path),
            "expected": {split: list(sequences) for split, sequences in assignment.items()},
        }

    config = {
        "experiment_name": "matrix",
        "strict_manifest": str(strict_path),
        "base_model_config": str(base_path),
        "folds": folds,
        "seeds": [20260712, 20260713, 20260714],
        "methods": list(PENN_NEURAL_METHODS),
        "strict_manifest_loading": True,
        "shared": {
            "epochs": 30,
            "static_context": {
                "window_size": 2048,
                "reference_scope": "train_sensor",
                "selection": "auto_low_motion",
            },
            "bias_auxiliary": {
                "enabled": False,
                "weight": 0.0,
                "stationary_fraction_threshold": 0.8,
            },
        },
        "fir": {"taps": 9, "ridge_alpha": 0.01},
        "output": {
            "runs_dir": str(tmp_path / "runs"),
            "active_run_file": str(tmp_path / "outputs/active_penn_matrix_run.txt"),
            "latest_dir": str(tmp_path / "outputs/latest"),
            "latest_run_file": str(tmp_path / "outputs/latest_run.txt"),
        },
    }

    def load_bundle(split_path: Path, _strict_path: Path) -> TrainingSplitBundle:
        return bundles[str(Path(split_path).resolve())]

    monkeypatch.setattr(penn_matrix_config, "load_training_split_manifest", load_bundle)
    return config, bundles, load_bundle


def _install_doubles(
    monkeypatch: pytest.MonkeyPatch,
    load_bundle: Any,
    *,
    checkpoint_mutator: Any = None,
    prediction_mutator: Any = None,
) -> list[dict[str, Any]]:
    from virtual_imu.training import penn_matrix_runner as runner

    calls: list[dict[str, Any]] = []

    def fake_train(config: dict[str, Any], config_path: Path | None = None) -> Path:
        del config_path
        calls.append(config)
        training_dir = Path(config["output"]["runs_dir"])
        training_dir.mkdir(parents=True, exist_ok=True)
        train_sequences = FOLD_ASSIGNMENTS[config["matrix_fold_id"]]["train"]
        checkpoint = {
            "config": config,
            "best_epoch": 1,
            "best_metric": 0.1,
            "checkpoint_metric": "val_score",
            "model_state_dict": {"weight": torch.ones(1)},
            "static_window_metadata": (
                {"low-a": {"sequence_id": train_sequences[0]}}
                if config["matrix_method"] not in {"tcn", "bias_only"}
                else {}
            ),
            "target_mode": config["model"]["target_mode"],
            "competitor_baseline": {},
        }
        if config["matrix_method"] == "fir_bias_drift_residual":
            bundle = load_bundle(Path(config["data"]["split_manifest"]), Path("unused"))
            checkpoint["competitor_baseline"] = {
                "enabled": True,
                "mode": "fir",
                "state": fit_system_id_baseline_state(
                    bundle.frames["train"], fir_taps=9, alpha=0.01
                ),
            }
        if checkpoint_mutator is not None:
            checkpoint_mutator(checkpoint, config)
        torch.save(checkpoint, training_dir / "best.ckpt")
        return training_dir

    class FakeGenerator:
        def __init__(self, checkpoint_path: Path, device: str = "auto") -> None:
            del device
            self.checkpoint = torch.load(checkpoint_path, map_location="cpu")

        def run_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
            result = frame.copy()
            if self.checkpoint["target_mode"] == "competitor_residual":
                values = predict_system_id_from_state(
                    self.checkpoint["competitor_baseline"]["state"], frame
                )["learned_fir_causal"]
            else:
                values = frame[HIGH].to_numpy(dtype=float)
            for index, destination in enumerate(VIRTUAL):
                result[destination] = values[:, index] + 0.001
            if prediction_mutator is not None:
                prediction_mutator(result, self.checkpoint)
            return result

    monkeypatch.setattr(runner, "load_training_split_manifest", load_bundle)
    monkeypatch.setattr(runner, "train_model", fake_train)
    monkeypatch.setattr(runner, "VirtualIMUGenerator", FakeGenerator)
    return calls


def _count_baseline_fits(monkeypatch: pytest.MonkeyPatch) -> list[pd.DataFrame]:
    from virtual_imu.training import penn_matrix_runner as runner

    original = runner.fit_penn_baselines
    calls: list[pd.DataFrame] = []

    def counted(train_df: pd.DataFrame, *args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append(train_df)
        return original(train_df, *args, **kwargs)

    monkeypatch.setattr(runner, "fit_penn_baselines", counted)
    return calls


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_matrix_cli_help_documents_safe_execution_modes() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_penn_loso_matrix.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--smoke" in result.stdout
    assert "--resume-run" in result.stdout


def test_smoke_matrix_writes_complete_artifacts_and_resumes_without_training(
    matrix_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, load_bundle = matrix_fixture
    calls = _install_doubles(monkeypatch, load_bundle)
    baseline_calls = _count_baseline_fits(monkeypatch)

    run_dir = run_penn_matrix(config, smoke=True)

    ledger = pd.read_csv(run_dir / "trial_ledger.csv")
    trials = pd.read_csv(run_dir / "trials.csv")
    assert len(calls) == 60
    assert len(baseline_calls) == 4
    assert len(ledger) == 60
    assert set(ledger["status"]) == {"complete"}
    assert len(list((run_dir / "baselines").glob("*/metrics/*.json"))) == 16
    assert len(trials) == 76
    assert (run_dir / "fold_summary.csv").is_file()
    assert (run_dir / "overall_summary.csv").is_file()
    assert (run_dir / "report.md").is_file()
    assert Path(config["output"]["active_run_file"]).read_text(encoding="utf-8").strip() == str(run_dir)
    assert Path(config["output"]["latest_run_file"]).read_text(encoding="utf-8").strip() == str(run_dir)

    resumed = run_penn_matrix(config, smoke=True, resume_run=run_dir)

    assert resumed == run_dir
    assert len(calls) == 60
    assert len(baseline_calls) == 4


def test_effective_snapshot_and_hash_are_exactly_the_trained_config(
    matrix_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, load_bundle = matrix_fixture
    calls = _install_doubles(monkeypatch, load_bundle)

    run_dir = run_penn_matrix(config, smoke=True)

    trained = {item["experiment_name"]: item for item in calls}
    for trial_json in (run_dir / "trials").glob("*/trial.json"):
        payload = json.loads(trial_json.read_text(encoding="utf-8"))
        snapshot = Path(payload["effective_config"])
        effective = load_config(snapshot)
        assert effective == trained[payload["trial_id"]]
        assert effective["output"]["runs_dir"] == str(
            (run_dir / "trials" / payload["trial_id"] / "training").resolve()
        )
        assert effective["matrix_fold_id"] == payload["fold_id"]
        assert effective["matrix_method"] == payload["method"]
        assert effective["seed"] == payload["seed"]
        assert payload["fingerprints"]["effective_config_sha256"] == _canonical_hash(
            effective
        )
        assert snapshot == run_dir / "configs/effective" / f"{payload['trial_id']}.yaml"


@pytest.mark.parametrize("source_kind", ["base", "split", "strict"])
def test_resume_refuses_changed_run_inputs_before_any_work(
    matrix_fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
) -> None:
    config, _, load_bundle = matrix_fixture
    train_calls = _install_doubles(monkeypatch, load_bundle)
    baseline_calls = _count_baseline_fits(monkeypatch)
    run_dir = run_penn_matrix(config, smoke=True)
    train_count = len(train_calls)
    fit_count = len(baseline_calls)

    if source_kind == "base":
        base = load_config(config["base_model_config"])
        base["model"]["dropout"] = 0.2
        save_config(base, config["base_model_config"])
    elif source_kind == "split":
        split_path = Path(config["folds"]["test_af"]["manifest"])
        split_path.write_text('{"fold_id": "test_af", "changed": true}', encoding="utf-8")
    else:
        Path(config["strict_manifest"]).write_text(
            '{"records": [], "changed": true}', encoding="utf-8"
        )

    with pytest.raises(ValueError, match="run fingerprint"):
        run_penn_matrix(config, smoke=True, resume_run=run_dir)
    assert len(train_calls) == train_count
    assert len(baseline_calls) == fit_count


def test_resume_refuses_corrupt_completed_baseline_before_any_work(
    matrix_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, load_bundle = matrix_fixture
    train_calls = _install_doubles(monkeypatch, load_bundle)
    baseline_calls = _count_baseline_fits(monkeypatch)
    run_dir = run_penn_matrix(config, smoke=True)
    train_count = len(train_calls)
    fit_count = len(baseline_calls)
    state_path = run_dir / "baselines/test_af/state.json"
    state_path.write_text(state_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="baseline artifact hash mismatch"):
        run_penn_matrix(config, smoke=True, resume_run=run_dir)
    assert len(train_calls) == train_count
    assert len(baseline_calls) == fit_count


def test_resume_rejects_completed_trial_with_changed_effective_config_hash(
    matrix_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, load_bundle = matrix_fixture
    _install_doubles(monkeypatch, load_bundle)
    run_dir = run_penn_matrix(config, smoke=True)
    effective_path = next((run_dir / "configs/effective").glob("*.yaml"))
    effective = effective_path.read_text(encoding="utf-8")
    effective_path.write_text(effective.replace("epochs: 1", "epochs: 2"), encoding="utf-8")

    with pytest.raises(ValueError, match="config hash mismatch"):
        run_penn_matrix(config, smoke=True, resume_run=run_dir)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("wrong_test", "held-out"),
        ("nonfinite", "finite"),
        ("missing_checkpoint", "checkpoint"),
        ("context_outside_train", "train split"),
        ("hybrid_missing_state", "FIR state"),
        ("hybrid_wrong_state", "train split"),
        ("hybrid_wrong_mode", "mode=fir"),
    ],
)
def test_trial_audit_rejects_invalid_artifacts(
    matrix_fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    expected: str,
) -> None:
    config, _, load_bundle = matrix_fixture

    def checkpoint_mutator(checkpoint: dict[str, Any], effective: dict[str, Any]) -> None:
        if kind == "missing_checkpoint" and effective["matrix_method"] == "tcn":
            return
        if kind == "context_outside_train" and effective["matrix_method"] == "drift_only":
            checkpoint["static_window_metadata"] = {"low-a": {"sequence_id": "outside"}}
        if kind == "hybrid_missing_state" and effective["matrix_method"] == "fir_bias_drift_residual":
            checkpoint["competitor_baseline"].pop("state")
        if kind == "hybrid_wrong_state" and effective["matrix_method"] == "fir_bias_drift_residual":
            checkpoint["competitor_baseline"]["state"]["fir_bias"][0] += 1.0
        if kind == "hybrid_wrong_mode" and effective["matrix_method"] == "fir_bias_drift_residual":
            checkpoint["competitor_baseline"]["mode"] = "blend"

    calls = _install_doubles(
        monkeypatch,
        load_bundle,
        checkpoint_mutator=checkpoint_mutator,
        prediction_mutator=(
            (lambda frame, checkpoint: frame.__setitem__("sequence_id", "wrong"))
            if kind == "wrong_test"
            else (
                (lambda frame, checkpoint: frame.__setitem__(VIRTUAL[0], np.inf))
                if kind == "nonfinite"
                else None
            )
        ),
    )
    if kind == "missing_checkpoint":
        from virtual_imu.training import penn_matrix_runner as runner

        original = runner.train_model

        def without_checkpoint(config: dict[str, Any], config_path: Path | None = None) -> Path:
            path = original(config, config_path)
            if config["matrix_method"] == "tcn":
                (path / "best.ckpt").unlink()
            return path

        monkeypatch.setattr(runner, "train_model", without_checkpoint)

    with pytest.raises((ValueError, FileNotFoundError, RuntimeError), match=expected):
        run_penn_matrix(config, smoke=True)
    assert calls
