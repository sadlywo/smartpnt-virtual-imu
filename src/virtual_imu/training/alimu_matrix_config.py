"""Validation and deterministic expansion of the ALiMU leave-sensor matrix."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from virtual_imu.config import load_config
from virtual_imu.training.dataset import FEATURE_COLUMNS

ALIMU_NEURAL_METHODS = (
    "tcn",
    "bias_only",
    "drift_only",
    "bias_drift",
    "fir_bias_drift_residual",
)
ALIMU_SEEDS = (20260712, 20260713, 20260714)
ALIMU_HOLDOUT_SENSOR = "MPU9250"
ALIMU_TRAIN_SENSORS = {"BMX160", "LSM9DS1"}
SHARED_STATIC_CONTEXT = {
    "window_size": 2048,
    "selection": "auto_low_motion",
    "reference_scope": "train_global",
}
SHARED_BIAS_AUXILIARY = {
    "enabled": False,
    "weight": 0.0,
    "stationary_fraction_threshold": 0.8,
}
REQUIRED_COLUMNS = {
    "timestamp_s",
    "gx_low",
    "gy_low",
    "gz_low",
    "ax_low",
    "ay_low",
    "az_low",
    "gx_high",
    "gy_high",
    "gz_high",
    "ax_high",
    "ay_high",
    "az_high",
    "temperature_c",
    "sequence_id",
    "low_sensor",
}


@dataclass(frozen=True)
class ALiMUTrialSpec:
    trial_id: str
    method: str
    seed: int
    holdout_sensor: str


def build_alimu_trial_specs(
    config: dict[str, Any], project_root: Path
) -> tuple[ALiMUTrialSpec, ...]:
    """Validate the single ALiMU leave-sensor split and expand its 15 trials."""

    return validate_alimu_matrix(config, project_root)


def validate_alimu_matrix(
    config: dict[str, Any], project_root: Path
) -> tuple[ALiMUTrialSpec, ...]:
    """Validate the ALiMU matrix contract and return its deterministic trial specs."""

    root = Path(project_root).resolve()
    _validate_matrix_axes(config)
    split_paths = _validate_split_paths(config, root)
    _validate_manifest(split_paths["manifest_path"])
    _validate_split_frames(split_paths)
    _validate_base_config(config, root, split_paths)

    trials = tuple(
        ALiMUTrialSpec(
            trial_id=f"mpu9250__{method}__seed_{seed}",
            method=method,
            seed=seed,
            holdout_sensor=ALIMU_HOLDOUT_SENSOR,
        )
        for method in ALIMU_NEURAL_METHODS
        for seed in ALIMU_SEEDS
    )
    if len(trials) != 15 or len({trial.trial_id for trial in trials}) != 15:
        raise ValueError("ALiMU matrix must produce 15 unique deterministic trial IDs")
    return trials


def build_alimu_effective_trial_config(
    matrix_config: dict[str, Any],
    base_config: dict[str, Any],
    trial: ALiMUTrialSpec,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Apply one controlled matrix ablation to an ALiMU base configuration."""

    if trial.method not in ALIMU_NEURAL_METHODS:
        raise ValueError(f"Unsupported ALiMU method: {trial.method}")
    if trial.seed not in ALIMU_SEEDS:
        raise ValueError(f"Unsupported ALiMU seed: {trial.seed}")
    if trial.holdout_sensor != ALIMU_HOLDOUT_SENSOR:
        raise ValueError(f"ALiMU holdout_sensor must be {ALIMU_HOLDOUT_SENSOR}")
    expected_trial_id = f"mpu9250__{trial.method}__seed_{trial.seed}"
    if trial.trial_id != expected_trial_id:
        raise ValueError(f"ALiMU trial_id must be exactly {expected_trial_id}")

    root = Path.cwd().resolve() if project_root is None else Path(project_root).resolve()
    validated_trials = validate_alimu_matrix(matrix_config, root)
    if trial not in validated_trials:
        raise ValueError("ALiMU trial must belong to the validated matrix contract")
    split_paths = _validate_split_paths(matrix_config, root)
    _validate_base_payload(base_config, split_paths)
    shared = _validate_shared_contract(matrix_config)

    effective = deepcopy(base_config)
    data = _require_mapping(effective, "data", "base config data")
    model = _require_mapping(effective, "model", "base config model")
    training = _require_mapping(effective, "training", "base config training")
    effective["seed"] = trial.seed
    effective["experiment_name"] = trial.trial_id
    for key in ("train_path", "val_path", "test_path", "manifest_path"):
        data[key] = str(split_paths[key])
    training["epochs"] = shared["epochs"]
    training["bias_auxiliary"] = deepcopy(shared["bias_auxiliary"])

    model["target_mode"] = "residual"
    model.pop("zero_init_head", None)
    training.pop("competitor_baseline", None)
    training.pop("baseline_margin_weight", None)
    training.pop("correction_energy_weight", None)

    if trial.method == "tcn":
        model["type"] = "tcn"
        for key in ("static_channels", "use_bias_branch", "use_drift_branch", "bias_branch_mode"):
            model.pop(key, None)
        data.pop("static_context", None)
    else:
        model["type"] = "bias_drift_tcn"
        model["use_bias_branch"] = trial.method != "drift_only"
        model["use_drift_branch"] = trial.method != "bias_only"
        if trial.method == "bias_only":
            model.pop("static_channels", None)
            data.pop("static_context", None)
        else:
            model["static_channels"] = 7
            data["static_context"] = {"enabled": True, **shared["static_context"]}

    if trial.method == "fir_bias_drift_residual":
        fir = _require_mapping(matrix_config, "fir", "fir")
        model["target_mode"] = "competitor_residual"
        model["zero_init_head"] = True
        training["competitor_baseline"] = {
            "enabled": True,
            "mode": "fir",
            "fir_taps": int(fir["taps"]),
            "ridge_alpha": float(fir["ridge_alpha"]),
        }
        training["baseline_margin_weight"] = 0.0
        training["correction_energy_weight"] = 0.0
    return effective


def build_effective_alimu_trial_config(
    matrix_config: dict[str, Any],
    base_config: dict[str, Any],
    trial: ALiMUTrialSpec,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias for :func:`build_alimu_effective_trial_config`."""

    return build_alimu_effective_trial_config(
        matrix_config, base_config, trial, project_root=project_root
    )


def _validate_matrix_axes(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise ValueError("ALiMU matrix config must be a mapping")
    if config.get("strict_manifest_loading") is not True:
        raise ValueError("strict_manifest_loading must be true for the ALiMU matrix")
    if config.get("holdout_sensor") != ALIMU_HOLDOUT_SENSOR:
        raise ValueError(f"holdout_sensor must be exactly {ALIMU_HOLDOUT_SENSOR}")
    if config.get("methods") != list(ALIMU_NEURAL_METHODS):
        raise ValueError(f"ALiMU matrix methods must be exactly {list(ALIMU_NEURAL_METHODS)}")
    if config.get("seeds") != list(ALIMU_SEEDS):
        raise ValueError(f"ALiMU matrix seeds must be exactly {list(ALIMU_SEEDS)} with no duplicates")
    _validate_shared_contract(config)
    fir = _require_mapping(config, "fir", "fir")
    if fir.get("taps") != 9 or fir.get("ridge_alpha") != 0.01:
        raise ValueError("ALiMU matrix FIR must use 9 taps and ridge alpha 0.01")
    output = _require_mapping(config, "output", "output")
    required_outputs = {"runs_dir", "active_run_file", "latest_dir", "latest_run_file"}
    if any(not str(output.get(key, "")).strip() for key in required_outputs):
        raise ValueError(f"ALiMU matrix output must configure {sorted(required_outputs)}")


def _validate_shared_contract(config: dict[str, Any]) -> dict[str, Any]:
    shared = _require_mapping(config, "shared", "shared")
    if shared.get("epochs") != 30:
        raise ValueError("shared.epochs must be exactly 30")
    static = _require_mapping(shared, "static_context", "shared.static_context")
    if static != SHARED_STATIC_CONTEXT:
        raise ValueError(f"shared.static_context must be exactly {SHARED_STATIC_CONTEXT}")
    bias_auxiliary = _require_mapping(shared, "bias_auxiliary", "shared.bias_auxiliary")
    if bias_auxiliary != SHARED_BIAS_AUXILIARY:
        raise ValueError("shared bias auxiliary loss must be disabled")
    return shared


def _validate_split_paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    split = _require_mapping(config, "split", "split")
    expected_root = root / "data/processed/alimu_strict/leave_sensor_mpu9250"
    paths = {
        "train_path": _resolve_path(split.get("train_path"), root, "split.train_path"),
        "val_path": _resolve_path(split.get("val_path"), root, "split.val_path"),
        "test_path": _resolve_path(split.get("test_path"), root, "split.test_path"),
        "manifest_path": _resolve_path(split.get("manifest_path"), root, "split.manifest_path"),
    }
    expected = {
        "train_path": (expected_root / "train.parquet").resolve(),
        "val_path": (expected_root / "val.parquet").resolve(),
        "test_path": (expected_root / "test.parquet").resolve(),
        "manifest_path": (expected_root / "manifest.json").resolve(),
    }
    for key, expected_path in expected.items():
        if paths[key] != expected_path:
            raise ValueError(f"split.{key} must point to {expected_path}")
    return paths


def _validate_manifest(manifest_path: Path) -> None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to load ALiMU split manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("ALiMU split manifest must be a mapping")
    policy = _require_mapping(manifest, "split_policy", "manifest split_policy")
    if policy.get("mode") != "leave_sensor_out":
        raise ValueError("ALiMU manifest split_policy.mode must be leave_sensor_out")
    if policy.get("holdout") != ALIMU_HOLDOUT_SENSOR:
        raise ValueError("ALiMU manifest holdout must be MPU9250")
    if policy.get("unit") != "sequence_id":
        raise ValueError("ALiMU manifest split_policy.unit must be sequence_id")
    splits = _require_mapping(manifest, "splits", "manifest splits")
    expected_sensors = {
        "train": ALIMU_TRAIN_SENSORS,
        "val": ALIMU_TRAIN_SENSORS,
        "test": {ALIMU_HOLDOUT_SENSOR},
    }
    for split_name, sensors in expected_sensors.items():
        declared = _require_mapping(splits, split_name, f"manifest splits.{split_name}").get("low_sensor")
        if not isinstance(declared, list) or set(declared) != sensors:
            raise ValueError(f"ALiMU manifest {split_name} low_sensor does not match the contract")


def _validate_split_frames(paths: dict[str, Path]) -> None:
    expected_sensors = {
        "train_path": ("train", ALIMU_TRAIN_SENSORS),
        "val_path": ("validation", ALIMU_TRAIN_SENSORS),
        "test_path": ("test", {ALIMU_HOLDOUT_SENSOR}),
    }
    for key, (split_name, expected) in expected_sensors.items():
        try:
            frame = pd.read_parquet(paths[key])
        except Exception as exc:
            raise ValueError(
                f"Unable to read ALiMU {split_name} split parquet at {paths[key]}"
            ) from exc
        missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"ALiMU {split_name} split is missing required columns: {missing}")
        sensor_values = frame["low_sensor"]
        if sensor_values.isna().any():
            raise ValueError(f"ALiMU {split_name} low_sensor assignment contains missing values")
        sensors = set(sensor_values.astype(str))
        if sensors != expected:
            raise ValueError(f"ALiMU {split_name} low_sensor assignment must be {sorted(expected)}")


def _validate_base_config(config: dict[str, Any], root: Path, split_paths: dict[str, Path]) -> None:
    base_path = _resolve_path(config.get("base_model_config"), root, "base_model_config")
    _validate_base_payload(load_config(base_path), split_paths)


def _validate_base_payload(base: dict[str, Any], split_paths: dict[str, Path]) -> None:
    if not isinstance(base, dict):
        raise ValueError("base config must be a mapping")
    data = _require_mapping(base, "data", "base config data")
    model = _require_mapping(base, "model", "base config model")
    training = _require_mapping(base, "training", "base config training")
    if data.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("ALiMU base config must use the seven native feature channels")
    if data.get("window_size") != 128:
        raise ValueError("ALiMU base config must use window_size 128")
    for key in ("train_path", "val_path", "test_path"):
        if _resolve_path(data.get(key), split_paths[key].parents[4], f"base data.{key}") != split_paths[key]:
            raise ValueError(f"ALiMU base config data.{key} must match the strict split")
    static = _require_mapping(data, "static_context", "base config data.static_context")
    if static != {"enabled": True, **SHARED_STATIC_CONTEXT}:
        raise ValueError("ALiMU base config must match the shared static context contract")
    if model.get("in_channels") != 7 or model.get("static_channels") != 7:
        raise ValueError("ALiMU base config must use seven native feature channels")
    if training.get("epochs") != 30:
        raise ValueError("ALiMU matrix base config must use exactly 30 epochs")
    bias_auxiliary = _require_mapping(training, "bias_auxiliary", "base config training.bias_auxiliary")
    if bias_auxiliary != SHARED_BIAS_AUXILIARY:
        raise ValueError("ALiMU base config bias auxiliary loss must be disabled")


def _require_mapping(parent: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _resolve_path(value: Any, root: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()
