from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from virtual_imu.config import load_config
from virtual_imu.training.alimu_matrix_config import (
    ALIMU_NEURAL_METHODS,
    ALIMU_SEEDS,
    ALiMUTrialSpec,
    build_alimu_effective_trial_config,
    build_alimu_trial_specs,
    build_effective_alimu_trial_config,
)

LOW_COLUMNS = ["gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low"]
HIGH_COLUMNS = ["gx_high", "gy_high", "gz_high", "ax_high", "ay_high", "az_high"]
FEATURE_COLUMNS = [*LOW_COLUMNS, "temperature_c"]


def _with_null_low_sensor(frame: pd.DataFrame) -> pd.DataFrame:
    frame.loc[0, "low_sensor"] = None
    return frame


def test_constants_and_trial_matrix_are_exact_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames = _matrix_config(tmp_path)
    _patch_parquet_reads(monkeypatch, frames)

    trials = build_alimu_trial_specs(config, project_root=tmp_path)

    assert ALIMU_NEURAL_METHODS == (
        "tcn",
        "bias_only",
        "drift_only",
        "bias_drift",
        "fir_bias_drift_residual",
    )
    assert ALIMU_SEEDS == (20260712, 20260713, 20260714)
    assert len(trials) == 15
    assert len({trial.trial_id for trial in trials}) == 15
    assert {trial.holdout_sensor for trial in trials} == {"MPU9250"}
    assert trials[0].trial_id == "mpu9250__tcn__seed_20260712"
    assert trials[-1].trial_id == "mpu9250__fir_bias_drift_residual__seed_20260714"
    assert all(
        trial.trial_id == f"mpu9250__{trial.method}__seed_{trial.seed}" for trial in trials
    )


def test_trial_spec_is_immutable() -> None:
    trial = ALiMUTrialSpec("id", "tcn", 1, "MPU9250")

    with pytest.raises(AttributeError):
        trial.seed = 2  # type: ignore[misc]


def test_effective_config_rejects_a_trial_with_a_noncanonical_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames = _matrix_config(tmp_path)
    _patch_parquet_reads(monkeypatch, frames)
    base = _load_yaml(tmp_path / config["base_model_config"])
    trial = ALiMUTrialSpec("not-a-matrix-id", "tcn", ALIMU_SEEDS[0], "MPU9250")

    with pytest.raises(ValueError, match="trial_id"):
        build_alimu_effective_trial_config(config, base, trial, project_root=tmp_path)


@pytest.mark.parametrize(
    ("split", "mutate", "match"),
    [
        ("train", lambda frame: frame.drop(columns=["temperature_c"]), "train"),
        ("val", _with_null_low_sensor, "validation"),
        ("test", _with_null_low_sensor, "test"),
    ],
)
def test_effective_config_validates_the_strict_split_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    split: str,
    mutate: Any,
    match: str,
) -> None:
    config, frames = _matrix_config(tmp_path)
    frames[_split_path(tmp_path, split)] = mutate(frames[_split_path(tmp_path, split)].copy())
    _patch_parquet_reads(monkeypatch, frames)
    base = _load_yaml(tmp_path / config["base_model_config"])
    trial = ALiMUTrialSpec(
        "mpu9250__tcn__seed_20260712", "tcn", ALIMU_SEEDS[0], "MPU9250"
    )

    with pytest.raises(ValueError, match=match):
        build_alimu_effective_trial_config(config, base, trial, project_root=tmp_path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda cfg: cfg["methods"].append("unsupported"), "methods"),
        (lambda cfg: cfg["seeds"].append(cfg["seeds"][0]), "seeds"),
        (lambda cfg: cfg.__setitem__("holdout_sensor", "BMX160"), "holdout_sensor"),
    ],
)
def test_rejects_invalid_matrix_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Any, match: str
) -> None:
    config, frames = _matrix_config(tmp_path)
    _patch_parquet_reads(monkeypatch, frames)
    mutate(config)

    with pytest.raises(ValueError, match=match):
        build_alimu_trial_specs(config, project_root=tmp_path)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda cfg: cfg["fir"].__setitem__("taps", 8), "FIR"),
        (lambda cfg: cfg["fir"].__setitem__("ridge_alpha", 0.02), "FIR"),
        (lambda cfg: cfg["shared"].__setitem__("epochs", 29), "30"),
        (
            lambda cfg: cfg["shared"]["static_context"].__setitem__("window_size", 1024),
            "static_context",
        ),
        (
            lambda cfg: cfg["shared"]["bias_auxiliary"].__setitem__("enabled", True),
            "bias auxiliary",
        ),
    ],
)
def test_rejects_invalid_shared_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: Any, match: str
) -> None:
    config, frames = _matrix_config(tmp_path)
    _patch_parquet_reads(monkeypatch, frames)
    mutate(config)

    with pytest.raises(ValueError, match=match):
        build_alimu_trial_specs(config, project_root=tmp_path)


def test_rejects_base_config_that_does_not_preserve_alimu_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames = _matrix_config(tmp_path)
    _patch_parquet_reads(monkeypatch, frames)
    base_path = tmp_path / config["base_model_config"]
    base = _load_yaml(base_path)
    base["data"]["feature_columns"] = LOW_COLUMNS
    _write_yaml(base_path, base)

    with pytest.raises(ValueError, match="seven native feature channels"):
        build_alimu_trial_specs(config, project_root=tmp_path)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("train_path", "data/processed/alimu_strict/other/train.parquet", "train_path"),
        ("manifest_path", "data/processed/alimu_strict/other/manifest.json", "manifest_path"),
    ],
)
def test_rejects_noncanonical_leave_sensor_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
    match: str,
) -> None:
    config, frames = _matrix_config(tmp_path)
    _patch_parquet_reads(monkeypatch, frames)
    config["split"][key] = value

    with pytest.raises(ValueError, match=match):
        build_alimu_trial_specs(config, project_root=tmp_path)


def test_rejects_missing_required_split_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames = _matrix_config(tmp_path)
    frames[_split_path(tmp_path, "train")] = frames[_split_path(tmp_path, "train")].drop(
        columns=["temperature_c"]
    )
    _patch_parquet_reads(monkeypatch, frames)

    with pytest.raises(ValueError, match="temperature_c"):
        build_alimu_trial_specs(config, project_root=tmp_path)


def test_wraps_unreadable_parquet_errors_with_the_logical_split_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames = _matrix_config(tmp_path)
    _patch_parquet_reads(monkeypatch, frames)
    from virtual_imu.training import alimu_matrix_config

    train_path = _split_path(tmp_path, "train")
    monkeypatch.setattr(
        alimu_matrix_config.pd,
        "read_parquet",
        lambda path: (_ for _ in ()).throw(FileNotFoundError(path)),
    )

    with pytest.raises(ValueError, match=r"ALiMU train split.*train\.parquet") as exc_info:
        build_alimu_trial_specs(config, project_root=tmp_path)

    assert str(train_path) in str(exc_info.value)


@pytest.mark.parametrize(
    ("split", "sensor", "match"),
    [
        ("train", "MPU9250", "train"),
        ("val", "MPU9250", "validation"),
        ("test", "BMX160", "test"),
    ],
)
def test_rejects_invalid_sensor_assignments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    split: str,
    sensor: str,
    match: str,
) -> None:
    config, frames = _matrix_config(tmp_path)
    frames[_split_path(tmp_path, split)] = _frame(sensor)
    _patch_parquet_reads(monkeypatch, frames)

    with pytest.raises(ValueError, match=match):
        build_alimu_trial_specs(config, project_root=tmp_path)


@pytest.mark.parametrize(
    ("split", "match"),
    [("train", "train"), ("val", "validation"), ("test", "test")],
)
def test_rejects_null_low_sensor_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, split: str, match: str
) -> None:
    config, frames = _matrix_config(tmp_path)
    frame = frames[_split_path(tmp_path, split)].copy()
    frame.loc[0, "low_sensor"] = None
    frames[_split_path(tmp_path, split)] = frame
    _patch_parquet_reads(monkeypatch, frames)

    with pytest.raises(ValueError, match=match):
        build_alimu_trial_specs(config, project_root=tmp_path)


def test_rejects_manifest_without_the_leave_sensor_mpu9250_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames = _matrix_config(tmp_path)
    _patch_parquet_reads(monkeypatch, frames)
    manifest_path = tmp_path / config["split"]["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_policy"]["holdout"] = "BMX160"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"manifest.*holdout"):
        build_alimu_trial_specs(config, project_root=tmp_path)


def test_effective_configs_apply_only_method_specific_ablation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, frames = _matrix_config(tmp_path)
    _patch_parquet_reads(monkeypatch, frames)
    base = _load_yaml(tmp_path / config["base_model_config"])
    trials = build_alimu_trial_specs(config, project_root=tmp_path)
    selected = {trial.method: trial for trial in trials if trial.seed == ALIMU_SEEDS[0]}
    untouched = deepcopy(base)

    effective = {
        method: build_effective_alimu_trial_config(config, base, selected[method], project_root=tmp_path)
        for method in ALIMU_NEURAL_METHODS
    }

    assert base == untouched
    for trial_config in effective.values():
        assert trial_config["data"]["feature_columns"] == FEATURE_COLUMNS
        assert trial_config["data"]["window_size"] == 128
        assert trial_config["training"]["epochs"] == 30
        assert trial_config["training"]["bias_auxiliary"] == config["shared"]["bias_auxiliary"]
        assert trial_config["data"]["train_path"] == str(_split_path(tmp_path, "train"))
        assert trial_config["data"]["val_path"] == str(_split_path(tmp_path, "val"))
        assert trial_config["data"]["test_path"] == str(_split_path(tmp_path, "test"))
        assert trial_config["data"]["manifest_path"] == str(
            (tmp_path / config["split"]["manifest_path"]).resolve()
        )

    assert effective["tcn"]["model"]["type"] == "tcn"
    assert effective["tcn"]["model"]["target_mode"] == "residual"
    assert "static_context" not in effective["tcn"]["data"]

    bias = effective["bias_only"]
    assert (bias["model"]["use_bias_branch"], bias["model"]["use_drift_branch"]) == (True, False)
    assert "static_context" not in bias["data"]

    drift = effective["drift_only"]
    assert (drift["model"]["use_bias_branch"], drift["model"]["use_drift_branch"]) == (False, True)
    assert drift["data"]["static_context"] == {"enabled": True, **config["shared"]["static_context"]}
    assert drift["model"]["static_channels"] == 7

    combined = effective["bias_drift"]
    assert (combined["model"]["use_bias_branch"], combined["model"]["use_drift_branch"]) == (True, True)
    assert combined["model"]["target_mode"] == "residual"

    hybrid = effective["fir_bias_drift_residual"]
    assert hybrid["model"]["target_mode"] == "competitor_residual"
    assert hybrid["model"]["zero_init_head"] is True
    assert hybrid["training"]["competitor_baseline"] == {
        "enabled": True,
        "mode": "fir",
        "fir_taps": 9,
        "ridge_alpha": 0.01,
    }
    assert hybrid["training"]["baseline_margin_weight"] == 0.0
    assert hybrid["training"]["correction_energy_weight"] == 0.0


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_ALIMU_INTEGRATION") != "1",
    reason="set RUN_ALIMU_INTEGRATION=1 to validate the real ALiMU split data",
)
def test_real_matrix_and_base_configs_preserve_the_alimu_contract() -> None:
    project_root = Path(__file__).resolve().parents[1]
    matrix = load_config(project_root / "configs/alimu_leave_sensor_matrix_3seed.yaml")
    base = load_config(project_root / "configs/alimu_leave_sensor_bias_drift_base.yaml")

    trials = build_alimu_trial_specs(matrix, project_root=project_root)

    assert len(trials) == 15
    assert trials[0].trial_id == "mpu9250__tcn__seed_20260712"
    assert trials[-1].trial_id == "mpu9250__fir_bias_drift_residual__seed_20260714"
    assert base["training"]["epochs"] == 30
    assert base["data"]["feature_columns"] == FEATURE_COLUMNS
    assert base["data"]["static_context"] == {
        "enabled": True,
        "window_size": 2048,
        "selection": "auto_low_motion",
        "reference_scope": "train_global",
    }
    assert base["model"]["static_channels"] == 7


def _matrix_config(tmp_path: Path) -> tuple[dict[str, Any], dict[Path, pd.DataFrame]]:
    split_root = tmp_path / "data/processed/alimu_strict/leave_sensor_mpu9250"
    split_root.mkdir(parents=True)
    manifest_path = split_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "split_policy": {"mode": "leave_sensor_out", "holdout": "MPU9250", "unit": "sequence_id"},
                "splits": {
                    "train": {"low_sensor": ["BMX160", "LSM9DS1"]},
                    "val": {"low_sensor": ["BMX160", "LSM9DS1"]},
                    "test": {"low_sensor": ["MPU9250"]},
                },
            }
        ),
        encoding="utf-8",
    )
    base_path = tmp_path / "base.yaml"
    _write_yaml(
        base_path,
        {
            "seed": 7,
            "experiment_name": "base",
            "data": {
                "train_path": "data/processed/alimu_strict/leave_sensor_mpu9250/train.parquet",
                "val_path": "data/processed/alimu_strict/leave_sensor_mpu9250/val.parquet",
                "test_path": "data/processed/alimu_strict/leave_sensor_mpu9250/test.parquet",
                "feature_columns": FEATURE_COLUMNS,
                "window_size": 128,
                "stride": 64,
                "batch_size": 128,
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
                "loss": "huber",
                "bias_auxiliary": {
                    "enabled": False,
                    "weight": 0.0,
                    "stationary_fraction_threshold": 0.8,
                },
            },
        },
    )
    config = {
        "experiment_name": "alimu_leave_sensor_matrix",
        "holdout_sensor": "MPU9250",
        "base_model_config": base_path.name,
        "strict_manifest_loading": True,
        "split": {
            "train_path": "data/processed/alimu_strict/leave_sensor_mpu9250/train.parquet",
            "val_path": "data/processed/alimu_strict/leave_sensor_mpu9250/val.parquet",
            "test_path": "data/processed/alimu_strict/leave_sensor_mpu9250/test.parquet",
            "manifest_path": "data/processed/alimu_strict/leave_sensor_mpu9250/manifest.json",
        },
        "methods": list(ALIMU_NEURAL_METHODS),
        "seeds": list(ALIMU_SEEDS),
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
            "runs_dir": "runs",
            "active_run_file": "outputs/active_alimu_matrix_run.txt",
            "latest_dir": "outputs/latest",
            "latest_run_file": "outputs/latest_run.txt",
        },
    }
    return config, {
        _split_path(tmp_path, "train"): _frame("BMX160", "LSM9DS1"),
        _split_path(tmp_path, "val"): _frame("BMX160", "LSM9DS1"),
        _split_path(tmp_path, "test"): _frame("MPU9250"),
    }


def _frame(*sensors: str) -> pd.DataFrame:
    rows = []
    for index, sensor in enumerate(sensors):
        row: dict[str, object] = {
            "timestamp_s": float(index),
            "temperature_c": 25.0,
            "sequence_id": f"{sensor.lower()}_{index}",
            "low_sensor": sensor,
        }
        row.update({column: 1.0 for column in LOW_COLUMNS})
        row.update({column: 2.0 for column in HIGH_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def _split_path(tmp_path: Path, split: str) -> Path:
    return (tmp_path / f"data/processed/alimu_strict/leave_sensor_mpu9250/{split}.parquet").resolve()


def _patch_parquet_reads(monkeypatch: pytest.MonkeyPatch, frames: dict[Path, pd.DataFrame]) -> None:
    from virtual_imu.training import alimu_matrix_config

    monkeypatch.setattr(
        alimu_matrix_config.pd,
        "read_parquet",
        lambda path: frames[Path(path).resolve()].copy(),
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))
