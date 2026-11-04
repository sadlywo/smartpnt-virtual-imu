from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from virtual_imu.config import load_config
from virtual_imu.training.penn_matrix_config import (
    PennTrialSpec,
    build_effective_trial_config,
    build_trial_specs,
)

FOLD_ASSIGNMENTS = {
    "test_af": {"train": ["as", "bf"], "validation": ["bs"], "test": ["af"]},
    "test_as": {"train": ["bf", "bs"], "validation": ["af"], "test": ["as"]},
    "test_bf": {"train": ["af", "bs"], "validation": ["as"], "test": ["bf"]},
    "default_bs": {"train": ["af", "as"], "validation": ["bf"], "test": ["bs"]},
}
METHODS = (
    "tcn",
    "bias_only",
    "drift_only",
    "bias_drift",
    "fir_bias_drift_residual",
)
SEEDS = (20260712, 20260713, 20260714)


def test_matrix_expands_to_sixty_paired_trials(tmp_path: Path) -> None:
    config = _matrix_config(tmp_path)

    trials = build_trial_specs(config, project_root=tmp_path)

    assert len(trials) == 60
    assert {trial.fold_id for trial in trials} == set(FOLD_ASSIGNMENTS)
    assert {trial.method for trial in trials} == set(METHODS)
    assert {trial.seed for trial in trials} == set(SEEDS)
    assert len({trial.trial_id for trial in trials}) == 60
    assert trials[0].trial_id == "test_af__tcn__seed_20260712"
    assert trials[-1].trial_id == "default_bs__fir_bias_drift_residual__seed_20260714"
    assert all(
        trial.trial_id == f"{trial.fold_id}__{trial.method}__seed_{trial.seed}"
        for trial in trials
    )


def test_trial_spec_is_immutable() -> None:
    trial = PennTrialSpec("id", "fold", Path("split.json"), "af", "tcn", 1)

    with pytest.raises(AttributeError):
        trial.seed = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda cfg: cfg["folds"].pop("test_af"), "four folds"),
        (lambda cfg: cfg["seeds"].append(cfg["seeds"][0]), "seeds"),
        (lambda cfg: cfg["methods"].append("unsupported"), "methods"),
    ],
)
def test_rejects_invalid_matrix_axes(tmp_path: Path, mutate: Any, match: str) -> None:
    config = _matrix_config(tmp_path)
    mutate(config)

    with pytest.raises(ValueError, match=match):
        build_trial_specs(config, project_root=tmp_path)


def test_rejects_non_thirty_epoch_base(tmp_path: Path) -> None:
    config = _matrix_config(tmp_path)
    base_path = tmp_path / config["base_model_config"]
    base = load_config(base_path)
    base["training"]["epochs"] = 29
    _write_yaml(base_path, base)

    with pytest.raises(ValueError, match="30 epochs"):
        build_trial_specs(config, project_root=tmp_path)


def test_rejects_disabled_strict_manifest_loading(tmp_path: Path) -> None:
    config = _matrix_config(tmp_path)
    config["strict_manifest_loading"] = False

    with pytest.raises(ValueError, match=r"strict_manifest_loading.*true"):
        build_trial_specs(config, project_root=tmp_path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda shared: shared.__setitem__("epochs", 29),
        lambda shared: shared["static_context"].__setitem__("window_size", 1024),
        lambda shared: shared["static_context"].__setitem__("reference_scope", "validation"),
        lambda shared: shared["static_context"].__setitem__("selection", "first"),
        lambda shared: shared["bias_auxiliary"].__setitem__("enabled", True),
    ],
)
def test_rejects_shared_contract_mismatch(tmp_path: Path, mutate: Any) -> None:
    config = _matrix_config(tmp_path)
    mutate(config["shared"])

    with pytest.raises(ValueError, match="shared"):
        build_trial_specs(config, project_root=tmp_path)


@pytest.mark.parametrize(
    ("section", "value", "match"),
    [
        ("shared", None, r"shared.*mapping"),
        ("shared.static_context", [], r"shared\.static_context.*mapping"),
        ("shared.bias_auxiliary", None, r"shared\.bias_auxiliary.*mapping"),
        ("output", [], r"output.*mapping"),
    ],
)
def test_rejects_malformed_matrix_nested_mappings(
    tmp_path: Path, section: str, value: Any, match: str
) -> None:
    config = _matrix_config(tmp_path)
    _set_nested(config, section, value)

    with pytest.raises(ValueError, match=match):
        build_trial_specs(config, project_root=tmp_path)


@pytest.mark.parametrize(
    ("section", "value", "match"),
    [
        ("training", None, r"base config training.*mapping"),
        ("model", [], r"base config model.*mapping"),
        ("data", None, r"base config data.*mapping"),
        ("data.static_context", [], r"base config data\.static_context.*mapping"),
        ("training.bias_auxiliary", None, r"base config training\.bias_auxiliary.*mapping"),
    ],
)
def test_rejects_malformed_base_nested_mappings(
    tmp_path: Path, section: str, value: Any, match: str
) -> None:
    config = _matrix_config(tmp_path)
    base_path = tmp_path / config["base_model_config"]
    base = load_config(base_path)
    _set_nested(base, section, value)
    _write_yaml(base_path, base)

    with pytest.raises(ValueError, match=match):
        build_trial_specs(config, project_root=tmp_path)


def test_rejects_manifest_with_wrong_assignment(tmp_path: Path) -> None:
    config = _matrix_config(tmp_path)
    split_path = tmp_path / config["folds"]["test_af"]["manifest"]
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    payload["splits"]["test"], payload["splits"]["validation"] = (
        payload["splits"]["validation"],
        payload["splits"]["test"],
    )
    split_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"test_af.*assignment"):
        build_trial_specs(config, project_root=tmp_path)


def test_effective_configs_apply_only_method_specific_contracts(tmp_path: Path) -> None:
    config = _matrix_config(tmp_path)
    base = load_config(tmp_path / config["base_model_config"])
    trials = build_trial_specs(config, project_root=tmp_path)
    by_method = {trial.method: trial for trial in trials if trial.fold_id == "test_af" and trial.seed == SEEDS[0]}
    base["data"]["static_context"].update(
        {"window_size": 64, "reference_scope": "test_sensor", "selection": "first"}
    )
    base["training"]["bias_auxiliary"] = {"enabled": True, "weight": 99.0}
    untouched = deepcopy(base)

    effective = {
        method: build_effective_trial_config(
            config, base, by_method[method], project_root=tmp_path
        )
        for method in METHODS
    }

    assert base == untouched
    shared_model = ("hidden_channels", "levels", "kernel_size", "dropout")
    shared_data = ("feature_columns", "window_size", "stride", "batch_size", "num_workers")
    for trial_config in effective.values():
        assert trial_config["training"]["epochs"] == 30
        assert trial_config["training"]["bias_auxiliary"] == config["shared"]["bias_auxiliary"]
        assert trial_config["seed"] == SEEDS[0]
        split_manifest = Path(trial_config["data"]["split_manifest"])
        strict_manifest = Path(trial_config["data"]["strict_manifest"])
        assert split_manifest.is_absolute()
        assert strict_manifest.is_absolute()
        assert split_manifest == (tmp_path / "test_af.json").resolve()
        assert strict_manifest == (tmp_path / "strict.json").resolve()
        assert all(trial_config["model"][key] == base["model"][key] for key in shared_model)
        assert all(trial_config["data"][key] == base["data"][key] for key in shared_data)

    assert effective["tcn"]["model"]["type"] == "tcn"
    assert effective["tcn"]["model"]["target_mode"] == "residual"
    assert "static_context" not in effective["tcn"]["data"]

    bias = effective["bias_only"]
    assert (bias["model"]["use_bias_branch"], bias["model"]["use_drift_branch"]) == (True, False)
    assert "static_context" not in bias["data"]

    drift = effective["drift_only"]
    assert (drift["model"]["use_bias_branch"], drift["model"]["use_drift_branch"]) == (False, True)
    assert drift["data"]["static_context"]["window_size"] == 2048
    assert drift["data"]["static_context"]["reference_scope"] == "train_sensor"
    assert drift["data"]["static_context"]["selection"] == "auto_low_motion"

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


def test_effective_configs_have_deep_copy_alias_isolation(tmp_path: Path) -> None:
    config = _matrix_config(tmp_path)
    base = load_config(tmp_path / config["base_model_config"])
    untouched = deepcopy(base)
    trials = build_trial_specs(config, project_root=tmp_path)
    selected = {
        trial.method: trial
        for trial in trials
        if trial.fold_id == "test_af" and trial.seed == SEEDS[0]
    }
    first = build_effective_trial_config(
        config, base, selected["drift_only"], project_root=tmp_path
    )
    second = build_effective_trial_config(
        config, base, selected["bias_drift"], project_root=tmp_path
    )

    first["data"]["feature_columns"].append("mutated")
    first["data"]["static_context"]["columns"].append("mutated")
    first["training"]["bias_auxiliary"]["weight"] = 42.0

    assert base == untouched
    assert "mutated" not in second["data"]["feature_columns"]
    assert "mutated" not in second["data"]["static_context"]["columns"]
    assert second["training"]["bias_auxiliary"]["weight"] == 0.0


def _matrix_config(tmp_path: Path) -> dict[str, Any]:
    strict_path = tmp_path / "strict.json"
    records = []
    split_records = {}
    for sequence in ("af", "as", "bf", "bs"):
        data_path = tmp_path / f"{sequence}.parquet"
        pd.DataFrame({"training_eligible": [True], "value": [1.0]}).to_parquet(
            data_path, index=False
        )
        identity = {
            "dataset_id": "penncosyvio",
            "pair_id": "tango_to_adis",
            "sequence_id": sequence,
            "fingerprint": f"fingerprint-{sequence}",
            "training_eligible": True,
        }
        records.append({**identity, "strict_path": str(data_path)})
        split_records[sequence] = {
            **identity,
            "path": str(data_path),
            "group_key": "sequence_id",
            "group_value": sequence,
            "row_filter": "all",
        }
    strict_path.write_text(json.dumps({"records": records}), encoding="utf-8")

    folds = {}
    for fold_id, assignments in FOLD_ASSIGNMENTS.items():
        split_path = tmp_path / f"{fold_id}.json"
        split_path.write_text(
            json.dumps(
                {
                    "fold_id": fold_id,
                    "split_kind": "leave_one_sequence_out",
                    "group_key": "sequence_id",
                    "splits": {
                        split: [split_records[sequence] for sequence in sequences]
                        for split, sequences in assignments.items()
                    },
                }
            ),
            encoding="utf-8",
        )
        folds[fold_id] = {"manifest": split_path.name, "expected": assignments}

    base_path = tmp_path / "base.yaml"
    _write_yaml(base_path, load_config("configs/penn_loso_bias_drift_tcn_default.yaml"))
    return {
        "experiment_name": "test_matrix",
        "strict_manifest": strict_path.name,
        "folds": folds,
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "base_model_config": base_path.name,
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
            "runs_dir": "runs",
            "active_run_file": "outputs/active_penn_matrix_run.txt",
            "latest_dir": "outputs/latest",
            "latest_run_file": "outputs/latest_run.txt",
        },
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _set_nested(payload: dict[str, Any], dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    target = payload
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
