from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from loguru import logger

from virtual_imu.config import save_config
from virtual_imu.eval.baselines import fit_system_id_baseline_state
from virtual_imu.preprocessing.normalization import compute_standardization
from virtual_imu.training.alimu_matrix_config import ALIMU_NEURAL_METHODS
from virtual_imu.training.alimu_matrix_runner import run_alimu_matrix
from virtual_imu.training.dataset import build_static_context_windows

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


def _sequence(sequence_id: str, sensor: str, offset: float) -> pd.DataFrame:
    timestamp = np.arange(24, dtype=float) / 40.0
    values = np.column_stack(
        [np.sin(timestamp * (axis + 1)) + offset + axis * 0.1 for axis in range(6)]
    )
    frame = pd.DataFrame(values, columns=LOW)
    for axis, column in enumerate(HIGH):
        frame[column] = values[:, axis] * (1.0 + axis * 0.01) - 0.02
    frame.insert(0, "timestamp_s", timestamp)
    frame["temperature_c"] = 25.0 + offset
    frame["sequence_id"] = sequence_id
    frame["low_sensor"] = sensor
    return frame


def _base_config() -> dict[str, Any]:
    return {
        "seed": 20260712,
        "experiment_name": "base",
        "data": {
            "train_path": "data/processed/alimu_strict/leave_sensor_mpu9250/train.parquet",
            "val_path": "data/processed/alimu_strict/leave_sensor_mpu9250/val.parquet",
            "test_path": "data/processed/alimu_strict/leave_sensor_mpu9250/test.parquet",
            "manifest_path": "data/processed/alimu_strict/leave_sensor_mpu9250/manifest.json",
            "feature_columns": [*LOW, "temperature_c"],
            "window_size": 128,
            "stride": 64,
            "batch_size": 8,
            "num_workers": 0,
            "static_context": {
                "enabled": True,
                "window_size": 2048,
                "selection": "auto_low_motion",
                "reference_scope": "train_global",
            },
        },
        "model": {
            "type": "bias_drift_tcn",
            "target_mode": "residual",
            "in_channels": 7,
            "out_channels": 6,
            "hidden_channels": 48,
            "levels": 4,
            "kernel_size": 5,
            "dropout": 0.05,
            "static_channels": 7,
            "use_bias_branch": True,
            "use_drift_branch": True,
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
    split_dir = tmp_path / "data/processed/alimu_strict/leave_sensor_mpu9250"
    split_dir.mkdir(parents=True)
    paths = {
        name: split_dir / f"{name}.parquet" for name in ("train", "val", "test")
    }
    for name, path in paths.items():
        path.write_bytes(f"parquet-bytes-{name}".encode())

    frames = {
        "train": pd.concat(
            [
                _sequence("train-bmx", "BMX160", 0.0),
                _sequence("train-lsm", "LSM9DS1", 0.2),
            ],
            ignore_index=True,
        ),
        "val": pd.concat(
            [
                _sequence("val-bmx", "BMX160", 0.4),
                _sequence("val-lsm", "LSM9DS1", 0.6),
            ],
            ignore_index=True,
        ),
        "test": _sequence("test-mpu", "MPU9250", 0.8),
    }
    manifest_path = split_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "split_policy": {
                    "mode": "leave_sensor_out",
                    "holdout": "MPU9250",
                    "unit": "sequence_id",
                },
                "splits": {
                    "train": {"low_sensor": ["BMX160", "LSM9DS1"]},
                    "val": {"low_sensor": ["BMX160", "LSM9DS1"]},
                    "test": {"low_sensor": ["MPU9250"]},
                },
            }
        ),
        encoding="utf-8",
    )
    base_path = tmp_path / "configs/base.yaml"
    save_config(_base_config(), base_path)
    output_dir = tmp_path / "outputs"
    config = {
        "experiment_name": "alimu_leave_sensor_matrix",
        "holdout_sensor": "MPU9250",
        "base_model_config": "configs/base.yaml",
        "strict_manifest_loading": True,
        "split": {
            "train_path": str(paths["train"].relative_to(tmp_path)),
            "val_path": str(paths["val"].relative_to(tmp_path)),
            "test_path": str(paths["test"].relative_to(tmp_path)),
            "manifest_path": str(manifest_path.relative_to(tmp_path)),
        },
        "methods": list(ALIMU_NEURAL_METHODS),
        "seeds": [20260712, 20260713, 20260714],
        "shared": {
            "epochs": 30,
            "static_context": {
                "window_size": 2048,
                "selection": "auto_low_motion",
                "reference_scope": "train_global",
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
            "active_run_file": str(output_dir / "active_alimu_matrix_run.txt"),
            "latest_dir": str(output_dir / "latest_alimu_matrix"),
            "latest_run_file": str(output_dir / "latest_alimu_matrix_run.txt"),
        },
    }

    original_read_parquet = pd.read_parquet

    def read_parquet(path: str | Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
        resolved = Path(path).resolve()
        for name, input_path in paths.items():
            if resolved == input_path.resolve():
                return frames[name].copy()
        return original_read_parquet(path, *args, **kwargs)

    from virtual_imu.training import alimu_matrix_config, alimu_matrix_runner

    monkeypatch.setattr(pd, "read_parquet", read_parquet)
    monkeypatch.setattr(alimu_matrix_config.pd, "read_parquet", read_parquet)
    monkeypatch.setattr(alimu_matrix_runner, "project_root_from_file", lambda _: tmp_path)
    return config, frames, paths


def _install_doubles(
    monkeypatch: pytest.MonkeyPatch,
    frames: dict[str, pd.DataFrame],
    *,
    checkpoint_mutator: Any = None,
    prediction_mutator: Any = None,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    from virtual_imu.training import alimu_matrix_runner as runner

    train_calls: list[dict[str, Any]] = []
    baseline_calls: list[pd.DataFrame] = []
    original_fit = runner.fit_penn_baselines
    feature_columns = list(_base_config()["data"]["feature_columns"])
    feature_stats = compute_standardization(frames["train"], feature_columns)

    def counted_fit(train: pd.DataFrame, *args: Any, **kwargs: Any) -> dict[str, Any]:
        baseline_calls.append(train.copy())
        return original_fit(train, *args, **kwargs)

    def fake_train(config: dict[str, Any], config_path: Path | None = None) -> Path:
        del config_path
        train_calls.append(config)
        training_dir = Path(config["output"]["runs_dir"])
        training_dir.mkdir(parents=True, exist_ok=True)
        method = config["matrix_method"]
        checkpoint: dict[str, Any] = {
            "config": deepcopy(config),
            "best_epoch": 1,
            "best_metric": 0.1,
            "checkpoint_metric": "val_score",
            "target_mode": config["model"]["target_mode"],
            "feature_columns": feature_columns,
            "feature_stats": feature_stats,
            "competitor_baseline": {},
            "static_context": deepcopy(config["data"].get("static_context", {})),
            "static_windows": {},
            "static_window_metadata": {},
        }
        if method in {"drift_only", "bias_drift", "fir_bias_drift_residual"}:
            metadata: dict[str, dict[str, Any]] = {}
            checkpoint["static_windows"] = build_static_context_windows(
                frames["train"],
                checkpoint["static_context"],
                feature_stats,
                metadata_out=metadata,
            )
            metadata["__train_global__"]["split"] = "train"
            checkpoint["static_window_metadata"] = metadata
        if method == "fir_bias_drift_residual":
            checkpoint["competitor_baseline"] = {
                "enabled": True,
                "mode": "fir",
                "state": fit_system_id_baseline_state(
                    frames["train"], fir_taps=9, alpha=0.01
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
            for index, destination in enumerate(VIRTUAL):
                result[destination] = result[HIGH[index]] + 0.001
            if prediction_mutator is not None:
                mutated = prediction_mutator(result, self.checkpoint)
                if mutated is not None:
                    result = mutated
            return result

    monkeypatch.setattr(runner, "fit_penn_baselines", counted_fit)
    monkeypatch.setattr(runner, "train_model", fake_train)
    monkeypatch.setattr(runner, "VirtualIMUGenerator", FakeGenerator)
    return train_calls, baseline_calls


def test_run_writes_fifteen_trial_ledger_nineteen_rows_and_resumes(
    matrix_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames, _ = matrix_fixture
    train_calls, baseline_calls = _install_doubles(monkeypatch, frames)

    run_dir = run_alimu_matrix(config, smoke=True)

    ledger = pd.read_csv(run_dir / "trial_ledger.csv")
    trials = pd.read_csv(run_dir / "trials.csv")
    assert len(ledger) == 15
    assert set(ledger["status"]) == {"complete"}
    assert len(trials) == 19
    assert len(train_calls) == 15
    assert len(baseline_calls) == 1
    assert set(baseline_calls[0]["low_sensor"]) == {"BMX160", "LSM9DS1"}
    assert len(list((run_dir / "baselines/metrics").glob("*.json"))) == 4
    baseline = json.loads((run_dir / "baselines/baseline.json").read_text(encoding="utf-8"))
    assert baseline["fit_split"] == "train"
    assert set(baseline["fit_sensors"]) == {"BMX160", "LSM9DS1"}
    assert len(baseline["state_hash"]) == 64
    assert set(baseline["prediction_hashes"]) == {
        "identity",
        "mean_bias",
        "channel_affine",
        "learned_fir_causal",
    }
    assert not (run_dir / "fold_summary.csv").exists()
    assert not (Path(config["output"]["latest_run_file"]).parent / "latest_run.txt").exists()
    assert Path(config["output"]["active_run_file"]).read_text(encoding="utf-8").strip() == str(run_dir)
    assert Path(config["output"]["latest_run_file"]).read_text(encoding="utf-8").strip() == str(run_dir)
    assert Path(config["output"]["latest_dir"]).is_dir()

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        resumed = run_alimu_matrix(config, smoke=True, resume_run=run_dir)
    finally:
        logger.remove(sink_id)

    assert resumed == run_dir
    assert len(train_calls) == 15
    assert len(baseline_calls) == 1
    assert sum("SKIPS kind=alimu_analytic_fit" in message for message in messages) == 1
    assert sum("SKIPS kind=alimu_neural_training" in message for message in messages) == 15


def test_resume_rejects_changed_parquet_bytes_before_work(
    matrix_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames, paths = matrix_fixture
    train_calls, baseline_calls = _install_doubles(monkeypatch, frames)
    run_dir = run_alimu_matrix(config, smoke=True)
    train_count = len(train_calls)
    fit_count = len(baseline_calls)
    paths["train"].write_bytes(b"changed-train-parquet-bytes")

    with pytest.raises(ValueError, match="run fingerprint"):
        run_alimu_matrix(config, smoke=True, resume_run=run_dir)

    assert len(train_calls) == train_count
    assert len(baseline_calls) == fit_count


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("wrong_sensor", "MPU9250"),
        ("nonfinite", "finite"),
        ("static_test_sensor", "training sensor"),
        ("static_missing_metadata", "metadata"),
        ("static_wrong_window", "static window"),
        ("static_nonfinite_window", "finite"),
        ("hybrid_missing_fir", "FIR state"),
        ("hybrid_wrong_state", "train split"),
        ("hybrid_wrong_hyperparams", "canonical FIR"),
    ],
)
def test_runner_rejects_prediction_and_checkpoint_protocol_violations(
    matrix_fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    expected: str,
) -> None:
    config, frames, _ = matrix_fixture

    def mutate_checkpoint(
        checkpoint: dict[str, Any], effective: dict[str, Any]
    ) -> None:
        method = effective["matrix_method"]
        if kind == "static_test_sensor" and method == "drift_only":
            checkpoint["static_window_metadata"]["__train_global__"].update(
                source_sensor="MPU9250", sequence_id="test-mpu"
            )
        if kind == "static_missing_metadata" and method == "drift_only":
            checkpoint["static_window_metadata"]["__train_global__"].pop("score")
        if kind == "static_wrong_window" and method == "drift_only":
            checkpoint["static_windows"]["__train_global__"][0, 0] += 1.0
        if kind == "static_nonfinite_window" and method == "drift_only":
            checkpoint["static_windows"]["__train_global__"][0, 0] = np.inf
        if kind == "hybrid_missing_fir" and method == "fir_bias_drift_residual":
            checkpoint["competitor_baseline"].pop("state")
        if kind == "hybrid_wrong_state" and method == "fir_bias_drift_residual":
            checkpoint["competitor_baseline"]["state"]["alpha"] = 0.02
        if kind == "hybrid_wrong_hyperparams" and method == "fir_bias_drift_residual":
            configured = checkpoint["config"]["training"]["competitor_baseline"]
            configured["fir_taps"] = 8
            checkpoint["competitor_baseline"]["state"] = (
                fit_system_id_baseline_state(frames["train"], fir_taps=8, alpha=0.01)
            )

    def mutate_prediction(
        prediction: pd.DataFrame, checkpoint: dict[str, Any]
    ) -> pd.DataFrame:
        del checkpoint
        if kind == "wrong_sensor":
            prediction["low_sensor"] = "BMX160"
        if kind == "nonfinite":
            prediction.loc[prediction.index[0], VIRTUAL[0]] = np.inf
        return prediction

    _install_doubles(
        monkeypatch,
        frames,
        checkpoint_mutator=mutate_checkpoint,
        prediction_mutator=mutate_prediction,
    )

    with pytest.raises(ValueError, match=expected):
        run_alimu_matrix(config, smoke=True)


def test_resume_rejects_tampered_input_snapshot_before_work(
    matrix_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames, _ = matrix_fixture
    train_calls, baseline_calls = _install_doubles(monkeypatch, frames)
    run_dir = run_alimu_matrix(config, smoke=True)
    train_count = len(train_calls)
    fit_count = len(baseline_calls)
    snapshot = run_dir / "configs/base.yaml"
    snapshot.write_text(snapshot.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="snapshot"):
        run_alimu_matrix(config, smoke=True, resume_run=run_dir)

    assert len(train_calls) == train_count
    assert len(baseline_calls) == fit_count


def test_resume_rejects_corrupt_completed_baseline_state(
    matrix_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames, _ = matrix_fixture
    train_calls, baseline_calls = _install_doubles(monkeypatch, frames)
    run_dir = run_alimu_matrix(config, smoke=True)
    train_count = len(train_calls)
    fit_count = len(baseline_calls)
    state = run_dir / "baselines/state.json"
    state.write_text(state.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash mismatch"):
        run_alimu_matrix(config, smoke=True, resume_run=run_dir)

    assert len(train_calls) == train_count
    assert len(baseline_calls) == fit_count


def test_resume_recomputes_all_top_level_baseline_audit_fields(
    matrix_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames, _ = matrix_fixture
    _install_doubles(monkeypatch, frames)
    run_dir = run_alimu_matrix(config, smoke=True)
    metadata_path = run_dir / "baselines/baseline.json"
    original = json.loads(metadata_path.read_text(encoding="utf-8"))

    mutations = {
        "fit_split": lambda payload: payload.update(fit_split="test"),
        "fit_sensors": lambda payload: payload.update(fit_sensors=["BMX160"]),
        "heldout_sensor": lambda payload: payload.update(heldout_sensor="BMX160"),
        "state_hash": lambda payload: payload.update(state_hash="0" * 64),
        "prediction_hashes": lambda payload: payload["prediction_hashes"].update(
            identity="0" * 64
        ),
    }
    for field, mutate in mutations.items():
        tampered = deepcopy(original)
        mutate(tampered)
        metadata_path.write_text(json.dumps(tampered, indent=2), encoding="utf-8")
        with pytest.raises(ValueError, match=field):
            run_alimu_matrix(config, smoke=True, resume_run=run_dir)

    metadata_path.write_text(json.dumps(original, indent=2), encoding="utf-8")


def test_resume_rejects_changed_matrix_base_and_manifest_inputs(
    matrix_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames, paths = matrix_fixture
    _install_doubles(monkeypatch, frames)
    run_dir = run_alimu_matrix(config, smoke=True)

    changed_matrix = deepcopy(config)
    changed_matrix["experiment_name"] = "changed_alimu_leave_sensor_matrix"
    with pytest.raises(ValueError, match="run fingerprint"):
        run_alimu_matrix(changed_matrix, smoke=True, resume_run=run_dir)

    base_path = Path(config["output"]["runs_dir"]).parent / "configs/base.yaml"
    original_base = base_path.read_text(encoding="utf-8")
    base_path.write_text(original_base + "\n# changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="run fingerprint"):
        run_alimu_matrix(config, smoke=True, resume_run=run_dir)
    base_path.write_text(original_base, encoding="utf-8")

    manifest_path = paths["train"].with_name("manifest.json")
    original_manifest = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(original_manifest)
    manifest["audit_change"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="run fingerprint"):
        run_alimu_matrix(config, smoke=True, resume_run=run_dir)
    manifest_path.write_text(original_manifest, encoding="utf-8")


def test_resume_rejects_missing_completed_checkpoint(
    matrix_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames, _ = matrix_fixture
    _install_doubles(monkeypatch, frames)
    run_dir = run_alimu_matrix(config, smoke=True)
    checkpoint = next((run_dir / "trials").glob("*/training/best.ckpt"))
    checkpoint.unlink()

    with pytest.raises(FileNotFoundError, match=r"checkpoint.*missing"):
        run_alimu_matrix(config, smoke=True, resume_run=run_dir)


def test_resume_audits_completed_aggregate_report_and_provenance_before_writing(
    matrix_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames, _ = matrix_fixture
    _install_doubles(monkeypatch, frames)
    run_dir = run_alimu_matrix(config, smoke=True)
    targets = {
        "aggregate": run_dir / "trials.csv",
        "report": run_dir / "report.md",
        "provenance": run_dir / "provenance.json",
    }

    for kind, path in targets.items():
        original = path.read_bytes()
        if kind == "provenance":
            payload = json.loads(original)
            payload["run_fingerprint"] = "0" * 64
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            path.write_bytes(original + b"corrupt")
        with pytest.raises(ValueError, match=kind):
            run_alimu_matrix(config, smoke=True, resume_run=run_dir)
        path.write_bytes(original)


def test_legacy_generic_latest_config_is_redirected_without_touching_penn(
    matrix_fixture: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames, _ = matrix_fixture
    _install_doubles(monkeypatch, frames)
    outputs = Path(config["output"]["latest_run_file"]).parent
    generic_latest = outputs / "latest"
    generic_latest.mkdir(parents=True)
    sentinel = generic_latest / "penn.txt"
    sentinel.write_text("keep-penn", encoding="utf-8")
    generic_pointer = outputs / "latest_run.txt"
    generic_pointer.write_text("penn-run\n", encoding="utf-8")
    config["output"]["latest_dir"] = str(generic_latest)
    config["output"]["latest_run_file"] = str(generic_pointer)

    run_dir = run_alimu_matrix(config, smoke=True)

    assert sentinel.read_text(encoding="utf-8") == "keep-penn"
    assert generic_pointer.read_text(encoding="utf-8") == "penn-run\n"
    assert (outputs / "latest_alimu_matrix/trials.csv").is_file()
    assert (outputs / "latest_alimu_matrix_run.txt").read_text(
        encoding="utf-8"
    ).strip() == str(run_dir)


def test_runner_rejects_escaped_latest_directory_without_deleting_it(
    matrix_fixture: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config, frames, _ = matrix_fixture
    _install_doubles(monkeypatch, frames)
    escaped_latest = tmp_path / "external/latest_alimu_matrix"
    escaped_latest.mkdir(parents=True)
    sentinel = escaped_latest / "do-not-delete.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    config["output"]["latest_dir"] = str(escaped_latest)

    with pytest.raises(ValueError, match="ALiMU output root"):
        run_alimu_matrix(config, smoke=True)

    assert sentinel.read_text(encoding="utf-8") == "preserve"
