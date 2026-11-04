"""Validation and deterministic expansion of the Penn LOSO trial matrix."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from virtual_imu.config import load_config
from virtual_imu.io.training_split_manifest import load_training_split_manifest

FOLD_ASSIGNMENTS = {
    "test_af": {"train": ("as", "bf"), "validation": ("bs",), "test": ("af",)},
    "test_as": {"train": ("bf", "bs"), "validation": ("af",), "test": ("as",)},
    "test_bf": {"train": ("af", "bs"), "validation": ("as",), "test": ("bf",)},
    "default_bs": {"train": ("af", "as"), "validation": ("bf",), "test": ("bs",)},
}
PENN_SEEDS = (20260712, 20260713, 20260714)
PENN_NEURAL_METHODS = (
    "tcn",
    "bias_only",
    "drift_only",
    "bias_drift",
    "fir_bias_drift_residual",
)
SHARED_STATIC_CONTEXT = {
    "window_size": 2048,
    "reference_scope": "train_sensor",
    "selection": "auto_low_motion",
}
SHARED_BIAS_AUXILIARY = {
    "enabled": False,
    "weight": 0.0,
    "stationary_fraction_threshold": 0.8,
}


@dataclass(frozen=True)
class PennTrialSpec:
    trial_id: str
    fold_id: str
    manifest_path: Path
    test_sequence: str
    method: str
    seed: int


def build_trial_specs(
    config: dict[str, Any], project_root: Path
) -> tuple[PennTrialSpec, ...]:
    """Validate the complete matrix and return its deterministic 60 trial specs."""

    root = Path(project_root).resolve()
    folds = _validate_matrix_axes(config)
    strict_path = _resolve_path(config["strict_manifest"], root, "strict_manifest")
    _validate_base_config(config, root)

    manifest_paths: dict[str, Path] = {}
    for fold_id, expected in FOLD_ASSIGNMENTS.items():
        fold = folds[fold_id]
        manifest_path = _resolve_path(fold["manifest"], root, f"folds.{fold_id}.manifest")
        if manifest_path.stem != fold_id:
            raise ValueError(f"folds.{fold_id}.manifest must identify {fold_id}.json")
        declared = _normalized_assignment(fold.get("expected"), f"folds.{fold_id}.expected")
        if declared != expected:
            raise ValueError(f"{fold_id} expected assignment does not match the Penn LOSO contract")

        bundle = load_training_split_manifest(manifest_path, strict_path)
        manifest_fold = str(bundle.split_manifest.get("fold_id", "")).strip()
        if manifest_fold != fold_id:
            raise ValueError(f"{fold_id} manifest declares fold_id {manifest_fold!r}")
        actual = {
            split: tuple(str(record.get("sequence_id", "")) for record in records)
            for split, records in bundle.split_manifest["splits"].items()
        }
        if actual != expected:
            raise ValueError(f"{fold_id} manifest assignment does not match the expected assignment")
        manifest_paths[fold_id] = manifest_path

    trials = tuple(
        PennTrialSpec(
            trial_id=f"{fold_id}__{method}__seed_{seed}",
            fold_id=fold_id,
            manifest_path=manifest_paths[fold_id],
            test_sequence=FOLD_ASSIGNMENTS[fold_id]["test"][0],
            method=method,
            seed=seed,
        )
        for fold_id in FOLD_ASSIGNMENTS
        for method in PENN_NEURAL_METHODS
        for seed in PENN_SEEDS
    )
    if len({trial.trial_id for trial in trials}) != 60:
        raise ValueError("Penn matrix must produce 60 unique deterministic trial IDs")
    return trials


def build_effective_trial_config(
    matrix_config: dict[str, Any],
    base_config: dict[str, Any],
    trial: PennTrialSpec,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Apply one method's controlled ablation to a copy of the shared base config."""

    if trial.method not in PENN_NEURAL_METHODS:
        raise ValueError(f"Unsupported Penn method: {trial.method}")
    _, _, base_training, _, _ = _base_sections(base_config)
    if int(base_training.get("epochs", 0)) != 30:
        raise ValueError("Penn matrix base config must use exactly 30 epochs")
    shared = _validate_shared_contract(matrix_config)
    root = Path.cwd().resolve() if project_root is None else Path(project_root).resolve()
    split_path = _resolve_path(trial.manifest_path, root, "trial manifest_path")
    strict_path = _resolve_path(matrix_config.get("strict_manifest"), root, "strict_manifest")

    effective = deepcopy(base_config)
    effective["seed"] = trial.seed
    effective["experiment_name"] = trial.trial_id
    data = _require_mapping(effective, "data", "base config data")
    model = _require_mapping(effective, "model", "base config model")
    training = _require_mapping(effective, "training", "base config training")
    data["split_manifest"] = str(split_path)
    data["strict_manifest"] = str(strict_path)
    training["epochs"] = shared["epochs"]
    training["bias_auxiliary"] = deepcopy(shared["bias_auxiliary"])

    model["target_mode"] = "residual"
    model.pop("zero_init_head", None)
    training.pop("competitor_baseline", None)
    training.pop("baseline_margin_weight", None)
    training.pop("correction_energy_weight", None)

    if trial.method == "tcn":
        model["type"] = "tcn"
        for key in ("static_channels", "use_bias_branch", "use_drift_branch"):
            model.pop(key, None)
        data.pop("static_context", None)
    else:
        model["type"] = "bias_drift_tcn"
        model["use_bias_branch"] = trial.method != "drift_only"
        model["use_drift_branch"] = trial.method != "bias_only"
        if trial.method == "bias_only":
            data.pop("static_context", None)
        else:
            static = data.setdefault("static_context", {})
            static.update({"enabled": True, **shared["static_context"]})

    if trial.method == "fir_bias_drift_residual":
        fir = matrix_config["fir"]
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


def _validate_matrix_axes(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not isinstance(config, dict):
        raise ValueError("Penn matrix config must be a mapping")
    if config.get("strict_manifest_loading") is not True:
        raise ValueError("strict_manifest_loading must be true for the Penn matrix")
    _validate_shared_contract(config)
    folds = config.get("folds")
    if not isinstance(folds, dict) or tuple(folds) != tuple(FOLD_ASSIGNMENTS):
        raise ValueError(f"Penn matrix must list exactly four folds: {list(FOLD_ASSIGNMENTS)}")
    if any(not isinstance(fold, dict) for fold in folds.values()):
        raise ValueError("Penn matrix fold definitions must be mappings")
    if config.get("seeds") != list(PENN_SEEDS):
        raise ValueError(f"Penn matrix seeds must be exactly {list(PENN_SEEDS)} with no duplicates")
    if config.get("methods") != list(PENN_NEURAL_METHODS):
        raise ValueError(f"Penn matrix methods must be exactly {list(PENN_NEURAL_METHODS)}")
    fir = _require_mapping(config, "fir", "fir")
    if fir.get("taps") != 9 or fir.get("ridge_alpha") != 0.01:
        raise ValueError("Penn matrix FIR must use 9 taps and ridge alpha 0.01")
    required_outputs = {"runs_dir", "active_run_file", "latest_dir", "latest_run_file"}
    output = _require_mapping(config, "output", "output")
    if any(not str(output.get(key, "")).strip() for key in required_outputs):
        raise ValueError(f"Penn matrix output must configure {sorted(required_outputs)}")
    return folds


def _validate_base_config(config: dict[str, Any], root: Path) -> None:
    base_path = _resolve_path(config.get("base_model_config"), root, "base_model_config")
    base = load_config(base_path)
    _, _, training, static, bias_auxiliary = _base_sections(base)
    if int(training.get("epochs", 0)) != 30:
        raise ValueError("Penn matrix base config must use exactly 30 epochs")
    if any(static.get(key) != value for key, value in SHARED_STATIC_CONTEXT.items()):
        raise ValueError("Penn matrix base config must match the shared static context contract")
    if any(bias_auxiliary.get(key) != value for key, value in SHARED_BIAS_AUXILIARY.items()):
        raise ValueError("Penn bias auxiliary loss must be disabled")


def _validate_shared_contract(config: dict[str, Any]) -> dict[str, Any]:
    shared = _require_mapping(config, "shared", "shared")
    if shared.get("epochs") != 30:
        raise ValueError("shared.epochs must be exactly 30")
    static = _require_mapping(shared, "static_context", "shared.static_context")
    if static != SHARED_STATIC_CONTEXT:
        raise ValueError(f"shared.static_context must be exactly {SHARED_STATIC_CONTEXT}")
    bias_auxiliary = _require_mapping(shared, "bias_auxiliary", "shared.bias_auxiliary")
    if bias_auxiliary != SHARED_BIAS_AUXILIARY:
        raise ValueError(f"shared.bias_auxiliary must be exactly {SHARED_BIAS_AUXILIARY}")
    return shared


def _base_sections(
    base: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not isinstance(base, dict):
        raise ValueError("base config must be a mapping")
    data = _require_mapping(base, "data", "base config data")
    model = _require_mapping(base, "model", "base config model")
    training = _require_mapping(base, "training", "base config training")
    static = _require_mapping(data, "static_context", "base config data.static_context")
    bias_auxiliary = _require_mapping(
        training, "bias_auxiliary", "base config training.bias_auxiliary"
    )
    return data, model, training, static, bias_auxiliary


def _require_mapping(parent: dict[str, Any], key: str, context: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _normalized_assignment(value: Any, context: str) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict) or set(value) != {"train", "validation", "test"}:
        raise ValueError(f"{context} must define train, validation, and test")
    normalized: dict[str, tuple[str, ...]] = {}
    for split in ("train", "validation", "test"):
        sequences = value[split]
        if not isinstance(sequences, list) or any(not isinstance(item, str) for item in sequences):
            raise ValueError(f"{context}.{split} must be a sequence list")
        normalized[split] = tuple(sequences)
    return normalized


def _resolve_path(value: Any, root: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()
