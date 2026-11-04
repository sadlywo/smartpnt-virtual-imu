"""Final-only, protocol-aware evaluation for the reviewer HPO experiments.

Classical parameters are fitted on train and selected on validation before
test prediction. Cache identities bind the physical train/validation rows,
manifests, frozen search grid and implementation. They deliberately exclude
test labels and neural seed/context so matched runs reuse one frozen fit.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger

from virtual_imu.eval.baselines import _sequence_lag_features
from virtual_imu.eval.formal_evaluation import (
    HIGH,
    IDENTITY,
    LOW,
    VIRTUAL,
    evaluate_prediction_frame,
    fit_pair_baselines,
    json_safe,
    physical_error_metrics,
    predict_complete_windows,
    predict_pair_baselines,
    verify_run_provenance,
    write_json,
)
from virtual_imu.eval.formal_navigation import (
    _physical_windows,
    _predict_checkpoint,
    _score_windows,
    _validate_stats,
    _verify_train_stats,
    fit_navigation_baselines,
    navigation_error_metrics,
    predict_navigation_baselines,
)
from virtual_imu.inference.virtual_imu_generator import VirtualIMUGenerator
from virtual_imu.io.training_bundle import sha256_file
from virtual_imu.io.training_split_manifest import load_training_split_manifest
from virtual_imu.models import make_model
from virtual_imu.training.gt_supervised_dataset import (
    SMARTPNT_GT_IMU_COLUMNS,
    SMARTPNT_GT_POSITION_COLUMNS,
    SMARTPNT_GT_POSITION_DELTA_COLUMNS,
    SmartPNTGTWindowDataset,
)
from virtual_imu.training.unified_data import (
    snapshot_unified_training_inputs,
    unified_navigation_frame,
)
from virtual_imu.utils.paths import project_root_from_file

DEFAULT_CLASSICAL_SEARCH = {
    "fit_split": "train",
    "select_split": "validation",
    "ridge_alpha": [1e-6, 1e-4, 0.01, 0.1, 1.0, 10.0, 100.0],
    "fir_taps": [1, 3, 5, 9, 17, 33, 65],
    "arx_taps": [3, 6, 12, 24, 48],
}


def _canonical(value) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _frame_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    ordered = frame.sort_values(["sequence_id", "timestamp_s"], kind="stable")
    values = pd.util.hash_pandas_object(ordered[columns], index=False).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _selection_score(prediction, reference, *, task: str, identity=None) -> float:
    if task == "navigation":
        return navigation_error_metrics(prediction, reference)["vector_rmse_m"]
    metrics = physical_error_metrics(prediction, reference)
    baseline = physical_error_metrics(identity, reference)
    return float(
        np.mean(
            [
                metrics[key] / max(baseline[key], 1e-12)
                for key in (
                    "gyro_vector_rmse_radps",
                    "accel_vector_rmse_mps2",
                )
            ]
        )
    )


def select_ridge_train_validation(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    alphas: list[float],
    *,
    task: str,
) -> dict:
    """Select an unpenalized-intercept ridge without accepting any test arrays.

    Paired designs start with the current six LOW channels; additional columns
    contain past LOW lags/differences only. Gram products are computed once
    per design, rather than once for every regularization candidate.
    """
    if task not in {"paired", "navigation"}:
        raise ValueError("Unknown ridge task")
    tx, ty, vx, vy = [
        np.asarray(value, dtype=np.float64)
        for value in (
            train_x,
            train_y,
            validation_x,
            validation_y,
        )
    ]
    outputs = 6 if task == "paired" else 3
    if (
        tx.ndim != 2
        or vx.ndim != 2
        or tx.shape[1] != vx.shape[1]
        or ty.shape != (len(tx), outputs)
        or vy.shape != (len(vx), outputs)
        or not len(tx)
        or not len(vx)
        or any(not np.isfinite(v).all() for v in (tx, ty, vx, vy))
    ):
        raise ValueError("Ridge selection requires finite matched train/validation arrays")
    if not alphas or any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError("Ridge search alphas must be finite and positive")
    train_design = np.column_stack([tx, np.ones(len(tx))])
    val_design = np.column_stack([vx, np.ones(len(vx))])
    gram, rhs = train_design.T @ train_design, train_design.T @ ty
    history, best = [], None
    penalty = np.eye(gram.shape[0])
    penalty[-1, -1] = 0
    for alpha in alphas:
        coefficients = np.linalg.solve(gram + float(alpha) * penalty, rhs)
        prediction = val_design @ coefficients
        score = _selection_score(
            prediction, vy, task=task, identity=vx[:, :6] if task == "paired" else None
        )
        if not np.isfinite(coefficients).all() or not np.isfinite(score):
            raise ValueError("Nonfinite classical candidate fit/selection score")
        history.append({"alpha": float(alpha), "validation_score": score})
        if best is None or score < best["validation_score"]:
            best = {
                "alpha": float(alpha),
                "validation_score": score,
                "weights": coefficients[:-1].tolist(),
                "bias": coefficients[-1].tolist(),
            }
    return {
        **best,
        "fit_split": "train",
        "selection_split": "validation",
        "fit_samples": len(tx),
        "selection_samples": len(vx),
        "intercept_penalized": False,
        "history": history,
    }


def _classical_location(run_dir: Path, config: dict) -> tuple[Path, dict]:
    for ancestor in run_dir.resolve().parents:
        matrix = ancestor / "matrix_config.yaml"
        if matrix.is_file():
            frozen = yaml.safe_load(matrix.read_text(encoding="utf-8"))
            return ancestor, copy.deepcopy(frozen.get("classical_search", DEFAULT_CLASSICAL_SEARCH))
    explicit = config["reviewer_protocol"].get("classical_search")
    if explicit is None:
        raise ValueError(
            "Final evaluation requires an ancestor frozen matrix_config.yaml or explicit classical_search"
        )
    return run_dir.parent, copy.deepcopy(explicit)


def _search_spec(search: dict) -> dict:
    result = {**DEFAULT_CLASSICAL_SEARCH, **search}
    if result["fit_split"] != "train" or result["select_split"] != "validation":
        raise ValueError("Classical search must fit train and select validation only")
    for name in ("ridge_alpha", "fir_taps", "arx_taps"):
        values = result[name]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"Classical search requires distinct nonempty {name}")
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError(f"Invalid classical search {name}")
        if name != "ridge_alpha" and any(type(v) is not int for v in values):
            raise ValueError("Lag taps must be positive integers")
    return result


def _cache_contract(task: str, config: dict, search: dict, payload: dict) -> dict:
    project = project_root_from_file(__file__)
    sources = {
        Path(__file__),
        Path(inspect.getfile(fit_pair_baselines)),
        Path(inspect.getfile(_sequence_lag_features)),
        Path(inspect.getfile(fit_navigation_baselines)),
    }
    manifests = {
        key: {"path": config["data"][key], "sha256": sha256_file(project / config["data"][key])}
        for key in ("dataset_manifest", "split_manifest", "strict_manifest", "assignment_manifest")
        if config["data"].get(key)
    }
    return {
        "schema_version": 1,
        "task": task,
        "protocol_id": config["reviewer_protocol"]["protocol_id"],
        "search": search,
        "manifests": manifests,
        "train_validation": payload,
        "implementation": {str(path.resolve()): sha256_file(path) for path in sorted(sources)},
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def _cached_classical(root: Path, contract: dict, fit) -> tuple[dict, dict]:
    identifier = contract["protocol_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError("Protocol id must be a plain path-safe slug")
    path = root / "classical" / identifier / f"{_digest(contract)}.json"
    reused = path.exists()
    if reused:
        logger.info("READS verified classical train/validation cache {}", path)
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("contract") != contract or cached.get("state_sha256") != _digest(
            cached["states"]
        ):
            raise ValueError("Classical cache contract or state checksum differs")
        logger.info("SKIPS classical refitting; fixed train/validation identity matches")
    else:
        logger.info("READS train/validation only for classical fits and selection {}", identifier)
        states = fit()
        cached = {"contract": contract, "states": states, "state_sha256": _digest(states)}
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("WRITES frozen classical cache {}", path)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(json_safe(cached), indent=2, allow_nan=False) + "\n")
    return cached["states"], {
        "path": str(path),
        "sha256": sha256_file(path),
        "contract_sha256": _digest(contract),
        "reused": reused,
        "fit_split": "train",
        "selection_split": "validation",
    }


def _fit_tapped(
    train: pd.DataFrame, validation: pd.DataFrame, taps: list[int], alphas, *, delta: bool
) -> dict:
    selected, history = None, []
    train_y, validation_y = train[HIGH].to_numpy(float), validation[HIGH].to_numpy(float)
    for count in taps:
        state = select_ridge_train_validation(
            _sequence_lag_features(train, count, include_delta=delta),
            train_y,
            _sequence_lag_features(validation, count, include_delta=delta),
            validation_y,
            alphas,
            task="paired",
        )
        history.extend({"taps": count, **row} for row in state["history"])
        if selected is None or state["validation_score"] < selected["validation_score"]:
            selected = {**state, "taps": count, "include_delta": delta}
    selected["history"] = history
    return selected


def _fit_paired_classical(train: pd.DataFrame, validation: pd.DataFrame, search: dict) -> dict:
    pairs = []
    validation_keys = set(map(tuple, validation[IDENTITY].astype(str).drop_duplicates().to_numpy()))
    for identity, group in train.groupby(IDENTITY, sort=True):
        identity = tuple(map(str, identity))
        target = validation
        for key, value in zip(IDENTITY, identity, strict=True):
            target = target[target[key].astype(str).eq(value)]
        if target.empty:
            raise ValueError("Classical device/reference mapping has no validation rows")
        pairs.append(
            {
                "identity": dict(zip(IDENTITY, identity, strict=True)),
                "fixed": fit_pair_baselines(group, fir_taps=9, alpha=0.01),
                "ridge_tuned": select_ridge_train_validation(
                    group[LOW].to_numpy(),
                    group[HIGH].to_numpy(),
                    target[LOW].to_numpy(),
                    target[HIGH].to_numpy(),
                    search["ridge_alpha"],
                    task="paired",
                ),
                "fir_tuned": _fit_tapped(
                    group, target, search["fir_taps"], search["ridge_alpha"], delta=False
                ),
                "arx_tuned": _fit_tapped(
                    group, target, search["arx_taps"], search["ridge_alpha"], delta=True
                ),
            }
        )
    if {tuple(pair["identity"][key] for key in IDENTITY) for pair in pairs} != validation_keys:
        raise ValueError(
            "Unseen validation device/reference mappings require a separately defined transfer protocol"
        )
    return {
        "fit_split": "train",
        "selection_split": "validation",
        "pairs": pairs,
        "selection_metric": "mean gyro/accel vector RMSE relative to identity on validation; separately per pair",
        "fixed_defaults": {"alpha": 0.01, "fir_taps": 9, "arx_taps": 12},
    }


def _predict_paired_classical(state: dict, frame: pd.DataFrame) -> dict:
    methods = predict_pair_baselines(state["fixed"], frame)
    for name in ("ridge_tuned", "fir_tuned", "arx_tuned"):
        fitted = state[name]
        features = (
            _sequence_lag_features(frame, fitted["taps"], include_delta=fitted["include_delta"])
            if "taps" in fitted
            else frame[LOW].to_numpy(float)
        )
        methods[name] = features @ np.asarray(fitted["weights"]) + np.asarray(fitted["bias"])
    return methods


def _pair_metrics(prediction, target) -> dict:
    metrics = physical_error_metrics(prediction, target)
    error = np.asarray(prediction) - np.asarray(target)
    for name, axis, unit in (("gyro", slice(0, 3), "radps"), ("accel", slice(3, 6), "mps2")):
        metrics[f"{name}_vector_mae_{unit}"] = float(np.linalg.norm(error[:, axis], axis=1).mean())
        metrics[f"{name}_bias_norm_{unit}"] = float(np.linalg.norm(error[:, axis].mean(axis=0)))
    return metrics


def _paired_frames(config: dict) -> tuple[dict, dict]:
    data = config["data"]
    bundle = load_training_split_manifest(
        Path(data["split_manifest"]), Path(data["strict_manifest"])
    )
    frames, skipped = {}, {}
    for split in ("train", "validation", "test"):
        pieces, omitted = [], []
        for name, group in bundle.frames[split].groupby("sequence_id", sort=False):
            group = group.sort_values("timestamp_s").reset_index(drop=True)
            if len(group[IDENTITY].drop_duplicates()) != 1:
                raise ValueError("Paired segments must have one device/reference identity")
            if len(group) < 320:
                omitted.append({"sequence_id": str(name), "samples": len(group)})
                logger.info(
                    "SKIPS short reviewer paired segment {}: {} <320 samples", name, len(group)
                )
                continue
            if not np.allclose(
                np.diff(group.timestamp_s.to_numpy(float)), 1 / 40, atol=1e-6, rtol=1e-5
            ):
                raise ValueError("Paired evaluation segment contains a gap or duplicate timestamp")
            if not np.isfinite(group[[*LOW, *HIGH]].to_numpy(float)).all():
                raise ValueError("Nonfinite paired physical measurements")
            pieces.append(group)
        if not pieces:
            raise ValueError(f"No common >=320-point paired segments in {split}")
        frames[split] = pd.concat(pieces, ignore_index=True)
        skipped[split] = omitted
    return frames, skipped


def _evaluate_paired(config, checkpoint, run_dir, device, root, search):
    frames, skipped = _paired_frames(config)
    generator = VirtualIMUGenerator(run_dir / "best.ckpt", device=str(device))
    validation, val_coverage = predict_complete_windows(
        generator, frames["validation"], batch_size=int(config["data"].get("batch_size", 128))
    )
    columns = [*IDENTITY, "sequence_id", "timestamp_s", *LOW, *HIGH]
    contract = _cache_contract(
        "paired",
        config,
        search,
        {
            "train_rows": len(frames["train"]),
            "validation_rows": len(validation),
            "train_sha256": _frame_digest(frames["train"], columns),
            "validation_sha256": _frame_digest(validation, columns),
            "coverage": "all complete stride40 windows; >=320-point sequences; contexts80/160/320 share covered rows",
        },
    )
    states, cache = _cached_classical(
        root, contract, lambda: _fit_paired_classical(frames["train"], validation, search)
    )
    test, test_coverage = predict_complete_windows(
        generator, frames["test"], batch_size=int(config["data"].get("batch_size", 128))
    )
    state_index = {
        tuple(pair["identity"][key] for key in IDENTITY): pair for pair in states["pairs"]
    }
    splits, tables, rows = {}, {}, []
    for split, frame, coverage in (
        ("validation", validation, val_coverage),
        ("test", test, test_coverage),
    ):
        evaluated = evaluate_prediction_frame(frame)
        evaluated.update(coverage=coverage, skipped_short_segments=skipped[split])
        table = frame.copy()
        for pair in evaluated["pairs"]:
            identity = tuple(str(pair[key]) for key in IDENTITY)
            if identity not in state_index:
                raise ValueError("Test device/reference mapping lacks a train-only baseline fit")
            mask = np.ones(len(frame), dtype=bool)
            for key, value in zip(IDENTITY, identity, strict=True):
                mask &= frame[key].astype(str).eq(value).to_numpy()
            group = frame.loc[mask]
            methods = {
                "model": group[VIRTUAL].to_numpy(),
                **_predict_paired_classical(state_index[identity], group),
            }
            pair["metrics"] = {
                name: _pair_metrics(values, group[HIGH].to_numpy())
                for name, values in methods.items()
            }
            pair["relative_scores"] = {
                name: _selection_score(
                    values, group[HIGH].to_numpy(), task="paired", identity=group[LOW].to_numpy()
                )
                for name, values in methods.items()
            }
            pair["relative_score"] = pair["relative_scores"]["model"]
            for name, values in methods.items():
                for index, axis in enumerate(("gx", "gy", "gz", "ax", "ay", "az")):
                    table.loc[mask, f"{name}_{axis}"] = values[:, index]
                metrics = pair["metrics"][name]
                rows.append(
                    {
                        "split": split,
                        **pair_identity(pair),
                        "method": name,
                        **{key: value for key, value in metrics.items() if np.isscalar(value)},
                        **{
                            f"{axis}_{label}_{'radps' if index < 3 else 'mps2'}": metrics[key][
                                index
                            ]
                            for index, axis in enumerate(("gx", "gy", "gz", "ax", "ay", "az"))
                            for key, label in (
                                ("axis_rmse", "rmse"),
                                ("axis_mae", "mae"),
                                ("axis_mean_error", "mean_error"),
                                ("axis_error_std", "error_std"),
                            )
                        },
                    }
                )
            pair["near_zero_identity"] = {
                key: pair["metrics"]["identity"][key] <= 1e-12
                for key in (
                    "gyro_vector_rmse_radps",
                    "accel_vector_rmse_mps2",
                )
            }
        datasets = sorted({pair["dataset_id"] for pair in evaluated["pairs"]})
        methods = evaluated["pairs"][0]["relative_scores"]
        evaluated["grouped_relative_scores"] = {
            name: float(
                np.mean(
                    [
                        np.mean(
                            [
                                pair["relative_scores"][name]
                                for pair in evaluated["pairs"]
                                if pair["dataset_id"] == dataset
                            ]
                        )
                        for dataset in datasets
                    ]
                )
            )
            for name in methods
        }
        evaluated["grouped_relative_score"] = evaluated["grouped_relative_scores"]["model"]
        tables[split], splits[split] = table, evaluated
    return (
        {
            "splits": splits,
            "scope": "Six LOW channels; dense physical residual correction; common >=320-point sequences and complete-window coverage; no streaming claim for noncausal models.",
        },
        tables,
        rows,
        states,
        cache,
    )


def pair_identity(pair: dict) -> dict:
    return {key: pair[key] for key in IDENTITY}


def _navigation_datasets(config, checkpoint):
    data = copy.deepcopy(config["data"])
    data.pop("load_splits", None)
    frame = unified_navigation_frame(data)
    roles = {
        "train": data["train_sequences"],
        "validation": data["val_sequences"],
        "test": data["test_sequences"],
    }
    names = [str(name) for values in roles.values() for name in values]
    if len(names) != len(set(names)) or set(frame.original_sequence_id.astype(str)) != set(names):
        raise ValueError(
            "Navigation original-sequence roles must be disjoint and match loaded records"
        )
    for name, group in frame.groupby("sequence_id", sort=False):
        if group.original_sequence_id.nunique() != 1:
            raise ValueError("Navigation segment spans original recordings")
        times = np.sort(group.timestamp_s.to_numpy(float))
        if not np.isfinite(
            group[[*SMARTPNT_GT_IMU_COLUMNS, *SMARTPNT_GT_POSITION_COLUMNS]].to_numpy(float)
        ).all() or not np.allclose(np.diff(times), 1 / 40, atol=1e-6, rtol=1e-5):
            raise ValueError(f"Nonfinite, gapped or duplicate navigation segment {name}")
    datasets, physical = {}, {}
    for split, originals in roles.items():
        selected = frame[frame.original_sequence_id.astype(str).isin(originals)].reset_index(
            drop=True
        )
        if "split" in selected and set(selected.split.astype(str)) != {split}:
            raise ValueError(
                "Navigation loaded split labels differ from configured recording assignment"
            )
        dataset = SmartPNTGTWindowDataset(
            selected,
            window_size=80,
            stride=40,
            label_mode="position_delta",
            feature_stats=checkpoint["feature_stats"],
            target_stats=checkpoint["target_stats"],
            normalize_features=bool(data.get("normalize_features", True)),
            normalize_targets=bool(data.get("normalize_targets", True)),
        )
        datasets[split], physical[split] = dataset, _physical_windows(dataset)
    _verify_train_stats(datasets["train"], physical["train"][1])
    return datasets, physical


def _fit_navigation_classical(train, validation, search):
    tx, ty, _ = train
    vx, vy, _ = validation
    history, best = [], None
    for alpha in search["ridge_alpha"]:
        state = fit_navigation_baselines(tx, ty, alpha=alpha)
        score = _selection_score(
            predict_navigation_baselines(state, vx)["ridge"], vy, task="navigation"
        )
        history.append({"alpha": alpha, "validation_score": score})
        if best is None or score < best["validation_score"]:
            best = {"state": state, "alpha": alpha, "validation_score": score}
    return {
        "fit_split": "train",
        "selection_split": "validation",
        "fixed": fit_navigation_baselines(tx, ty, alpha=0.01),
        "ridge_tuned": {**best, "history": history},
        "fit_windows": len(ty),
        "selection_windows": len(vy),
    }


def _evaluate_navigation(config, checkpoint, run_dir, device, root, search):
    datasets, physical = _navigation_datasets(config, checkpoint)
    payload = {}
    for split in ("train", "validation"):
        features, targets, metadata = physical[split]
        payload[split] = {
            "windows": len(targets),
            "features_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
            "targets_sha256": hashlib.sha256(targets.tobytes()).hexdigest(),
            "metadata_sha256": _digest(metadata.to_dict("records")),
        }
    contract = _cache_contract("navigation", config, search, payload)
    states, cache = _cached_classical(
        root,
        contract,
        lambda: _fit_navigation_classical(physical["train"], physical["validation"], search),
    )
    model = make_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    splits, tables, rows = {}, {}, []
    for split in ("validation", "test"):
        features, targets, metadata = physical[split]
        prediction = _predict_checkpoint(
            model, datasets[split], device, int(config["data"].get("batch_size", 128))
        )
        methods = {
            "model": prediction,
            **predict_navigation_baselines(states["fixed"], features),
            "ridge_tuned": predict_navigation_baselines(states["ridge_tuned"]["state"], features)[
                "ridge"
            ],
        }
        evaluated = _score_windows(metadata, methods, targets)
        table = metadata.copy()
        table["split"] = split
        for index, axis in enumerate(SMARTPNT_GT_POSITION_DELTA_COLUMNS):
            table[f"true_{axis}"] = targets[:, index]
            for name, values in methods.items():
                table[f"{'pred' if name == 'model' else name}_{axis}"] = values[:, index]
        for name, values in methods.items():
            evaluated["metrics"][name]["vector_mae_m"] = float(
                np.linalg.norm(values - targets, axis=1).mean()
            )
            evaluated["metrics"][name]["axis_error_std_m"] = (values - targets).std(axis=0).tolist()
            evaluated["metrics"][name]["bias_norm_m"] = float(
                np.linalg.norm((values - targets).mean(axis=0))
            )
            metrics = evaluated["metrics"][name]
            rows.append(
                {
                    "split": split,
                    "method": name,
                    **{key: value for key, value in metrics.items() if np.isscalar(value)},
                    **{
                        f"{axis}_{label}_m": metrics[key][index]
                        for index, axis in enumerate(("east", "north", "up"))
                        for key, label in (
                            ("axis_rmse_m", "rmse"),
                            ("axis_mae_m", "mae"),
                            ("axis_bias_m", "bias"),
                            ("axis_error_std_m", "error_std"),
                        )
                    },
                }
            )
        tables[split], splits[split] = table, evaluated
    return (
        {
            "splits": splits,
            "window_points": 80,
            "stride_points": 40,
            "sample_rate_hz": 40.0,
            "endpoint_span_s": 1.975,
            "axis_order": ["east", "north", "up"],
            "scope": "Rotated original-sequence assignments; six LOW IMU inputs; local ENU endpoint displacement over 1.975s, not trajectory ATE/RPE.",
        },
        tables,
        rows,
        states,
        cache,
    )


def _validate_final_checkpoint(task, config, checkpoint):
    protocol = config.get("reviewer_protocol", {})
    phase = protocol.get("phase")
    if phase not in {"final_default", "final_tuned"} or protocol.get(
        "variant"
    ) != phase.removeprefix("final_"):
        raise ValueError(
            "Reviewer evaluation is permitted only for frozen final_default/final_tuned jobs"
        )
    if protocol.get("selection_scope") != "validation_only":
        raise ValueError("Final models must have validation-only selection metadata")
    if checkpoint.get("config") != config or checkpoint.get("model_config") != config["model"]:
        raise ValueError("Checkpoint/config/model identity differs")
    weights = checkpoint.get("model_state_dict", {})
    if not weights or any(
        not torch.is_tensor(value) or not torch.isfinite(value).all().item()
        for value in weights.values()
    ):
        raise ValueError("Checkpoint requires finite model state tensors")
    data, model = config["data"], config["model"]
    if (
        int(model.get("in_channels", 0)) != 6
        or int(data.get("stride", 0)) != 40
        or float(data.get("rate_hz", 0)) != 40
    ):
        raise ValueError("Reviewer evaluation requires six LOW inputs, stride40 and40Hz")
    if checkpoint.get("checkpoint_metric") != config["training"]["checkpoint_metric"]:
        raise ValueError("Checkpoint selection metric differs from frozen config")
    if task == "paired":
        if (
            data.get("feature_columns") != LOW
            or checkpoint.get("feature_columns") != LOW
            or checkpoint.get("target_columns") != HIGH
            or checkpoint.get("target_mode") != "residual"
            or model.get("target_mode") != "residual"
            or model.get("out_channels") != 6
            or int(data["window_size"]) not in {80, 160, 320}
        ):
            raise ValueError(
                "Paired final evaluation requires matched six-channel residual windows80/160/320"
            )
        _validate_stats(checkpoint.get("feature_stats"), LOW, True, "feature_stats")
        _validate_stats(checkpoint.get("target_stats"), HIGH, True, "target_stats")
    else:
        expected_roles = {
            "train": data["train_sequences"],
            "val": data["val_sequences"],
            "test": data["test_sequences"],
        }
        if (
            int(data["window_size"]) != 80
            or model.get("out_channels") != 3
            or data.get("label_mode") != "position_delta"
            or data.get("input_columns", SMARTPNT_GT_IMU_COLUMNS) != SMARTPNT_GT_IMU_COLUMNS
            or model.get("target_layout") != "window"
            or model.get("prediction_head") != "last"
            or checkpoint.get("label_mode") != "position_delta"
            or checkpoint.get("target_layout") != "window"
            or checkpoint.get("prediction_head") != "last"
            or checkpoint.get("split_sequences") != expected_roles
        ):
            raise ValueError(
                "Navigation final evaluation requires frozen rotated roles, position_delta and last head"
            )
        if (
            checkpoint.get("feature_columns") != SMARTPNT_GT_IMU_COLUMNS
            or checkpoint.get("target_columns") != SMARTPNT_GT_POSITION_DELTA_COLUMNS
        ):
            raise ValueError(
                "Navigation checkpoint columns differ from physical six-input/ENU target order"
            )
        _validate_stats(
            checkpoint.get("feature_stats"),
            SMARTPNT_GT_IMU_COLUMNS,
            bool(data.get("normalize_features", True)),
            "feature_stats",
        )
        _validate_stats(
            checkpoint.get("target_stats"),
            SMARTPNT_GT_POSITION_DELTA_COLUMNS,
            bool(data.get("normalize_targets", True)),
            "target_stats",
        )


@contextmanager
def _math_inference():
    saved = (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
        torch.are_deterministic_algorithms_enabled(),
    )
    cudnn = (
        torch.backends.cuda.cudnn_sdp_enabled()
        if hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
        else None
    )
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    precision = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if cudnn is not None:
            torch.backends.cuda.enable_cudnn_sdp(False)
        with torch.inference_mode():
            yield
    finally:
        torch.backends.cuda.enable_flash_sdp(saved[0])
        torch.backends.cuda.enable_mem_efficient_sdp(saved[1])
        torch.backends.cuda.enable_math_sdp(saved[2])
        torch.use_deterministic_algorithms(saved[3])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = precision[:2]
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = precision[2:]
        if cudnn is not None:
            torch.backends.cuda.enable_cudnn_sdp(cudnn)
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def evaluate_reviewer_run(task: str, run_dir: Path, device: str = "cuda") -> dict:
    """Evaluate one frozen final job; search/promotion jobs cannot access test."""
    if task not in {"paired", "navigation"}:
        raise ValueError("Reviewer task must be paired or navigation")
    run_dir = Path(run_dir).resolve()
    evaluation = run_dir / "evaluation"
    if evaluation.exists():
        raise FileExistsError(f"Evaluation artifacts already exist: {evaluation}")
    logger.info("READS frozen reviewer config/checkpoint {}", run_dir)
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    checkpoint = torch.load(run_dir / "best.ckpt", map_location="cpu", weights_only=False)
    _validate_final_checkpoint(task, config, checkpoint)
    verify_run_provenance(run_dir, config, checkpoint["config"])
    root, search = _classical_location(run_dir, config)
    search = _search_spec(search)
    resolved = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    logger.info(
        "SKIPS test-driven candidate selection, target-input teacher forcing, test normalization fitting and evaluation-fitted alignment"
    )
    with _math_inference():
        evaluator = _evaluate_paired if task == "paired" else _evaluate_navigation
        body, tables, rows, states, cache = evaluator(
            config, checkpoint, run_dir, resolved, root, search
        )
    report = {
        "training_run": str(run_dir),
        "task": task,
        "reviewer_protocol": config["reviewer_protocol"],
        "seed": config["seed"],
        "selection": checkpoint["checkpoint_metric"],
        "best_epoch": checkpoint.get("best_epoch"),
        "best_metric": checkpoint.get("best_metric"),
        "classical_cache": cache,
        "baseline_state_file": "evaluation/baseline_train_state.json",
        "provenance_file": "evaluation/provenance.json",
        **body,
    }
    logger.info("WRITES immutable final reviewer evaluation {}", evaluation)
    evaluation.mkdir(exist_ok=False)
    frozen = snapshot_unified_training_inputs(config["data"], evaluation)
    training_provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    if frozen.get("unified_dataset") != training_provenance.get("unified_dataset"):
        raise ValueError("Final evaluation manifest provenance differs from training")
    provenance = {
        **frozen,
        "training_provenance": training_provenance,
        "classical_cache": cache,
        "checkpoint": {
            "path": str(run_dir / "best.ckpt"),
            "sha256": sha256_file(run_dir / "best.ckpt"),
            "git_commit": checkpoint.get("git_commit"),
        },
        "config": {
            "path": str(run_dir / "config.yaml"),
            "sha256": sha256_file(run_dir / "config.yaml"),
        },
        "evaluator_sha256": sha256_file(Path(__file__)),
        "device": str(resolved),
        "attention_backend": "math SDPA",
        "precision": "FP32, no AMP/TF32, strict determinism",
        "test_used_for_selection": False,
    }
    for name, value in (
        ("metrics.json", report),
        ("baseline_train_state.json", states),
        ("provenance.json", provenance),
    ):
        write_json(evaluation / name, value)
    for split, table in tables.items():
        logger.info("WRITES {} reviewer physical predictions {}", split, len(table))
        table.to_parquet(evaluation / f"{split}_predictions.parquet", index=False)
    pd.DataFrame(rows).to_csv(evaluation / f"{task}_metrics.csv", index=False)
    return json_safe(report)
