"""Postprocess frozen reviewer models on ALiMU<->Penn zero-shot transfer.

Full mode requires all 93 selections and 558 final evaluations to be complete.
Source checkpoint normalization/calibration is preserved. Target labels are
used only for final scoring; target-trained models are separately labelled
references, never selected by their target test performance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
)
from virtual_imu.eval.reviewer_evaluation import _math_inference, _pair_metrics, _search_spec
from virtual_imu.inference.virtual_imu_generator import VirtualIMUGenerator
from virtual_imu.io.training_split_manifest import load_training_split_manifest
from virtual_imu.training.reviewer_hpo import _verify_evaluation, _verify_selection
from virtual_imu.utils.paths import project_root_from_file

DIRECTIONS = {
    "alimu_to_penn": ("alimu_only", "penn_only", "alimu", "penncosyvio"),
    "penn_to_alimu": ("penn_only", "alimu_only", "penncosyvio", "alimu"),
}
PHYSICAL = [*IDENTITY, "sequence_id", "timestamp_s", *LOW, *HIGH]


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(json_safe(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def write_json_once(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )


def frame_fingerprint(frame: pd.DataFrame) -> str:
    ordered = frame.sort_values(["sequence_id", "timestamp_s"], kind="stable")
    return hashlib.sha256(
        pd.util.hash_pandas_object(ordered[PHYSICAL], index=False).to_numpy().tobytes()
    ).hexdigest()


def verify_record(directory: Path) -> dict:
    manifest = json.loads((directory / "artifact_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        path = directory / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Transfer artifact checksum differs: {path}")
    return manifest


def finish_record(staging: Path, destination: Path) -> None:
    files = {
        path.relative_to(staging).as_posix(): sha(path)
        for path in staging.rglob("*")
        if path.is_file()
    }
    write_json_once(staging / "artifact_manifest.json", {"files": files})
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)


def verify_pipeline(state: dict, root: Path, *, allow_subset: bool) -> list[dict]:
    """Reject test access until all declared selections are frozen."""
    cells = state.get("cells", {})
    finals = [
        job for job in state.get("jobs", {}).values() if job.get("phase", "").startswith("final_")
    ]
    complete = [
        job
        for job in finals
        if job.get("status") == "complete" and job.get("evaluation_status") == "complete"
    ]
    if not allow_subset and (
        state.get("status") != "complete"
        or state.get("smoke")
        or len(cells) != 93
        or len(finals) != 558
        or len(complete) != 558
    ):
        raise ValueError(
            "Full transfer requires a complete nonsmoke 93-cell/558-final reviewer pipeline"
        )
    if not cells or any("selected" not in cell for cell in cells.values()):
        raise ValueError("Every declared cell selection must be frozen before transfer")
    for identifier, cell in cells.items():
        selected = root / "selections" / f"{identifier.replace('/', '__')}.json"
        if (
            not selected.is_file()
            or sha(selected) != cell.get("selection_sha256")
            or json.loads(selected.read_text(encoding="utf-8")) != cell["selected"]
        ):
            raise ValueError("Frozen source selection checksum/identity differs")
        if not allow_subset:
            _verify_selection(identifier, state, root)
    seeds = state.get("final_seeds", [])
    variants = state.get("final_variants", [])
    if not allow_subset and (len(seeds) != 3 or set(variants) != {"default", "tuned"}):
        raise ValueError("Full transfer requires both variants and three final seeds")
    expected = {
        (cell, f"final_{variant}", seed) for cell in cells for variant in variants for seed in seeds
    }
    actual = [(job["cell"], job["phase"], job["seed"]) for job in complete]
    if len(actual) != len(set(actual)) or (not allow_subset and set(actual) != expected):
        raise ValueError("Final checkpoint identity coverage differs from the frozen matrix")
    if not complete:
        raise ValueError("No completed final jobs available even for explicit subset debugging")
    return complete


def _common_frame(frame: pd.DataFrame, *, trim_to_coverage: bool = True) -> pd.DataFrame:
    pieces = []
    for name, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values("timestamp_s").reset_index(drop=True)
        if len(group) < 320:
            print(f"SKIPS: target/source segment {name} has {len(group)} rows <320", flush=True)
            continue
        if not np.isfinite(group[LOW + HIGH].to_numpy()).all() or not np.allclose(
            np.diff(group.timestamp_s), 0.025, atol=1e-6, rtol=1e-5
        ):
            raise ValueError("Transfer segments require finite contiguous 40Hz physical rows")
        pieces.append(group.iloc[: len(group) // 40 * 40] if trim_to_coverage else group)
    if not pieces:
        raise ValueError("No common >=320-point transfer support")
    return pd.concat(pieces, ignore_index=True)


def _macro_score(prediction: np.ndarray, validation: pd.DataFrame) -> float:
    scores = defaultdict(list)
    for identity, positions in validation.groupby(IDENTITY, sort=True).indices.items():
        target = validation.iloc[positions][HIGH].to_numpy()
        measured = physical_error_metrics(prediction[positions], target)
        original = physical_error_metrics(validation.iloc[positions][LOW].to_numpy(), target)
        scores[str(identity[0])].append(
            float(
                np.mean(
                    [
                        measured[key] / max(original[key], 1e-12)
                        for key in (
                            "gyro_vector_rmse_radps",
                            "accel_vector_rmse_mps2",
                        )
                    ]
                )
            )
        )
    return float(np.mean([np.mean(values) for values in scores.values()]))


def _select_shared(train, validation, alphas, *, taps=None, delta=False):
    tx = (
        _sequence_lag_features(train, taps, include_delta=delta)
        if taps
        else train[LOW].to_numpy(float)
    )
    vx = (
        _sequence_lag_features(validation, taps, include_delta=delta)
        if taps
        else validation[LOW].to_numpy(float)
    )
    design = np.column_stack([tx, np.ones(len(tx))])
    validation_design = np.column_stack([vx, np.ones(len(vx))])
    gram, rhs = design.T @ design, design.T @ train[HIGH].to_numpy(float)
    penalty = np.eye(design.shape[1])
    penalty[-1, -1] = 0
    best, history = None, []
    for alpha in alphas:
        coefficients = np.linalg.solve(gram + alpha * penalty, rhs)
        score = _macro_score(validation_design @ coefficients, validation)
        if not np.isfinite(coefficients).all() or not np.isfinite(score):
            raise ValueError("Nonfinite shared source calibration fit")
        history.append({"alpha": alpha, "taps": taps, "validation_score": score})
        if best is None or score < best["validation_score"]:
            best = {
                "alpha": alpha,
                "taps": taps,
                "include_delta": delta,
                "validation_score": score,
                "weights": coefficients[:-1].tolist(),
                "bias": coefficients[-1].tolist(),
            }
    return {**best, "history": history}


def fit_source_classical(train: pd.DataFrame, validation: pd.DataFrame, search: dict) -> dict:
    """One domain-shared fit; no target arrays or target-device calibration API."""
    training = train.copy()
    training[IDENTITY] = "source_domain_shared"
    fixed = fit_pair_baselines(training, fir_taps=9, alpha=0.01)
    tuned = {"source_ridge_tuned": _select_shared(train, validation, search["ridge_alpha"])}
    for name, taps, delta in (
        ("source_fir_tuned", search["fir_taps"], False),
        ("source_arx_tuned", search["arx_taps"], True),
    ):
        candidates = [
            _select_shared(train, validation, search["ridge_alpha"], taps=count, delta=delta)
            for count in taps
        ]
        selected = min(candidates, key=lambda value: value["validation_score"])
        tuned[name] = {
            **selected,
            "history": [row for candidate in candidates for row in candidate["history"]],
        }
    return {
        "fit_split": "source_train",
        "selection_split": "source_validation",
        "fixed": fixed,
        "tuned": tuned,
        "fit_rows": len(train),
        "selection_rows": len(validation),
        "source_identities": train[IDENTITY].drop_duplicates().to_dict("records"),
        "selection_metric": "dataset-balanced mean of device-pair gyro/accel RMSE ratios relative to identity",
        "interpretation": "shared source-domain calibration applied without target calibration or target labels",
    }


def predict_source_classical(state: dict, target: pd.DataFrame) -> dict:
    result = {
        ("identity" if name == "identity" else f"source_{name}_fixed"): values
        for name, values in predict_pair_baselines(state["fixed"], target).items()
    }
    for name, fitted in state["tuned"].items():
        features = (
            _sequence_lag_features(target, fitted["taps"], include_delta=fitted["include_delta"])
            if fitted["taps"]
            else target[LOW].to_numpy(float)
        )
        result[name] = features @ np.asarray(fitted["weights"]) + np.asarray(fitted["bias"])
    return result


def score_transfer(frame: pd.DataFrame, methods: dict) -> dict:
    result = evaluate_prediction_frame(frame)
    for pair in result["pairs"]:
        mask = np.ones(len(frame), dtype=bool)
        for key in IDENTITY:
            mask &= frame[key].eq(pair[key]).to_numpy()
        target = frame.loc[mask, HIGH].to_numpy()
        pair["metrics"] = {
            name: _pair_metrics(values[mask], target) for name, values in methods.items()
        }
    result["grouped_relative_scores"] = {
        name: _macro_score(values, frame) for name, values in methods.items()
    }
    result["grouped_relative_score"] = result["grouped_relative_scores"]["model"]
    return result


def _processing_sources(project, state, *, allow_subset):
    executed = {
        str(path.relative_to(project)): sha(path)
        for path in (project / "src/virtual_imu").rglob("*.py")
    }
    frozen = state.get("contract", {}).get("source_sha256", {})
    matched = {}
    for path, expected in frozen.items():
        normalized = str(path).replace("\\", "/")
        if "/src/virtual_imu/" not in normalized:
            continue
        relative = "src/virtual_imu/" + normalized.split("/src/virtual_imu/", 1)[1]
        if executed.get(relative) != expected:
            raise ValueError(f"Executed source differs from frozen training source: {relative}")
        matched[relative] = expected
    if not matched and not allow_subset:
        raise ValueError("No frozen training implementation hashes available")
    return {
        "standalone_script_sha256": sha(Path(__file__)),
        "matched_trained_source_sha256": matched,
        "executed_source_sha256": executed,
    }


def _load_direction(index, direction, project):
    spec = index["protocols"][direction]
    split_path, strict_path = [Path(spec[key]) for key in ("split_manifest", "strict_manifest")]
    split_path = split_path if split_path.is_absolute() else project / split_path
    strict_path = strict_path if strict_path.is_absolute() else project / strict_path
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if (
        split.get("canonical_provenance", {}).get("dataset_manifest_sha256")
        != index["base_dataset_sha256"]
    ):
        raise ValueError("Transfer protocol canonical dataset hash differs")
    source_protocol, target_protocol = DIRECTIONS[direction][:2]
    role_sources = {
        "train": source_protocol,
        "validation": source_protocol,
        "test": target_protocol,
    }
    reference_manifests = {}
    for role, protocol in role_sources.items():
        reference_path = Path(index["protocols"][protocol]["split_manifest"])
        reference_path = (
            reference_path if reference_path.is_absolute() else project / reference_path
        )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))

        def identity(row):
            return (row["sequence_id"], row["path"], row["sha256"], row.get("row_filter", "all"))

        if {identity(row) for row in split["splits"][role]} != {
            identity(row) for row in reference["splits"][role]
        }:
            raise ValueError(
                "Transfer source/target role differs from the corresponding trained in-domain protocol"
            )
        reference_manifests[protocol] = {"path": str(reference_path), "sha256": sha(reference_path)}
    bundle = load_training_split_manifest(
        split_path, strict_path, split_names=["train", "validation", "test"]
    )
    source_id, target_id = DIRECTIONS[direction][2:]
    for role, expected in (("train", source_id), ("validation", source_id), ("test", target_id)):
        if set(bundle.frames[role].dataset_id.astype(str)) != {expected}:
            raise ValueError("Transfer role contains the wrong source/target dataset")
    frames = {
        role: _common_frame(frame, trim_to_coverage=role != "train")
        for role, frame in bundle.frames.items()
    }
    return frames, {
        "split_manifest": str(split_path),
        "split_sha256": sha(split_path),
        "strict_manifest": str(strict_path),
        "strict_sha256": sha(strict_path),
        "corresponding_in_domain_manifests": reference_manifests,
        "physical_fingerprints": {role: frame_fingerprint(frame) for role, frame in frames.items()},
    }


def _staging(out, label):
    path = out / "_partial" / f"{label}_{uuid.uuid4().hex[:12]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _classical_record(out, direction, frames, search, input_contract):
    path = out / "classical" / direction
    expected = {"inputs": input_contract, "search": search}
    if path.exists():
        verify_record(path)
        saved = json.loads((path / "state.json").read_text(encoding="utf-8"))
        if saved["contract"] != expected:
            raise ValueError("Source classical cache contract changed")
        return saved["state"], path
    state = fit_source_classical(frames["train"], frames["validation"], search)
    predictions = predict_source_classical(state, frames["test"])
    table = frames["test"][[*PHYSICAL, "original_sequence_id"]].copy()
    for name, values in predictions.items():
        for axis, column in enumerate(LOW):
            table[f"{name}_{column.removesuffix('_low')}"] = values[:, axis]
    stage = _staging(out, direction + "_classical")
    write_json_once(stage / "state.json", {"contract": expected, "state": state})
    table.to_parquet(stage / "target_predictions.parquet", index=False)
    finish_record(stage, path)
    return state, path


def _source_job(job, state, project):
    directory = Path(job["run_dir"])
    directory = directory if directory.is_absolute() else project / directory
    if (
        sha(directory / "best.ckpt") != job["checkpoint_sha256"]
        or sha(directory / "config.yaml") != job["run_config_sha256"]
    ):
        raise ValueError("Frozen source checkpoint/config checksum differs")
    config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    protocol = config["reviewer_protocol"]
    if (
        protocol["phase"] != job["phase"]
        or protocol.get("protocol_id") != job["cell"].split("/", 1)[0]
        or protocol.get("model_family") != job["cell"].split("/", 1)[1]
        or config["seed"] != job["seed"]
        or config["data"]["feature_columns"] != LOW
        or config["model"]["in_channels"] != 6
        or config["model"]["out_channels"] != 6
    ):
        raise ValueError("Source checkpoint is not the matched six-channel final paired job")
    if (
        job["phase"] == "final_tuned"
        and protocol["parameters"] != state["cells"][job["cell"]]["selected"]["trial"]["parameters"]
    ):
        raise ValueError("Source tuned parameters differ from the frozen validation winner")
    return directory, config


def _reference(job, state, root, target_frame, project):
    directory, config = _source_job(job, state, project)
    _verify_evaluation(job, root)
    saved = pd.read_parquet(directory / "evaluation/test_predictions.parquet", columns=PHYSICAL)
    if frame_fingerprint(saved) != frame_fingerprint(target_frame):
        raise ValueError("Target-trained reference and transfer do not share physical test rows")
    path = directory / "evaluation/metrics.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "label": "target_domain_trained_reference_not_zero_shot",
        "checkpoint_sha256": job["checkpoint_sha256"],
        "training_run": str(directory),
        "metrics_path": str(path),
        "metrics_sha256": sha(path),
        "selection_scope": config["reviewer_protocol"]["selection_scope"],
        "test": report["splits"]["test"],
    }


def _aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["direction"], record["model_family"], record["variant"])].append(record)
    rows = []
    for (direction, family, variant), entries in sorted(groups.items()):
        for identity in entries[0]["metrics"]["pairs"]:
            pair_id = tuple(identity[key] for key in IDENTITY)
            selected = [
                next(
                    pair
                    for pair in entry["metrics"]["pairs"]
                    if tuple(pair[key] for key in IDENTITY) == pair_id
                )["metrics"]["model"]
                for entry in entries
            ]
            row = {
                "direction": direction,
                "model_family": family,
                "variant": variant,
                **dict(zip(IDENTITY, pair_id, strict=True)),
                "seeds": len(entries),
                "scope": "optimization-seed mean/sampleSD; not independent-recording CI",
            }
            for key in selected[0]:
                if key != "samples" and np.isscalar(selected[0][key]):
                    values = [item[key] for item in selected]
                    row[key + "_mean"] = float(np.mean(values))
                    row[key + "_sample_sd"] = (
                        float(np.std(values, ddof=1)) if len(values) > 1 else None
                    )
                    references = [
                        entry["target_domain_reference"].get("test", {}).get("pairs", [])
                        for entry in entries
                    ]
                    if all(references):
                        reference_values = [
                            next(
                                pair
                                for pair in pairs
                                if tuple(pair[column] for column in IDENTITY) == pair_id
                            )["metrics"]["model"][key]
                            for pairs in references
                        ]
                        row["target_trained_reference_" + key + "_mean"] = float(
                            np.mean(reference_values)
                        )
                        row["transfer_minus_target_trained_" + key + "_mean"] = float(
                            np.mean(np.asarray(values) - reference_values)
                        )
            rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--protocol-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-completed-subset", action="store_true", help="Explicit NONFINAL debugging only"
    )
    args = parser.parse_args(argv)
    project = project_root_from_file(__file__)
    pipeline = args.pipeline_root.resolve()
    print(
        f"READS: frozen reviewer pipeline {pipeline}, protocol index {args.protocol_index}",
        flush=True,
    )
    print(f"WRITES: source-only cross-domain predictions/metrics under {args.out}", flush=True)
    print(
        "SKIPS: target normalization/calibration fitting, target validation, model/test-based selection, source mutation",
        flush=True,
    )
    state = json.loads((pipeline / "pipeline.json").read_text(encoding="utf-8"))
    finals = verify_pipeline(state, pipeline, allow_subset=args.allow_completed_subset)
    index = json.loads(args.protocol_index.read_text(encoding="utf-8"))
    if index["base_dataset_sha256"] != state["dataset_manifest_sha256"]:
        raise ValueError("Protocol index and trained pipeline bind different canonical datasets")
    source_hashes = _processing_sources(project, state, allow_subset=args.allow_completed_subset)
    lookup = {(job["cell"], job["phase"], job["seed"]): job for job in finals}
    jobs = []
    for direction, (source, target, _, _) in DIRECTIONS.items():
        for job in finals:
            protocol, family = job["cell"].split("/", 1)
            reference = lookup.get((f"{target}/{family}", job["phase"], job["seed"]))
            if protocol == source and reference is not None:
                jobs.append((direction, family, job, reference))
    if not jobs or (not args.allow_completed_subset and len(jobs) != 120):
        raise ValueError("Full transfer requires exactly 120 source-model evaluations")
    contract = {
        "pipeline": str(pipeline),
        "pipeline_sha256": sha(pipeline / "pipeline.json"),
        "protocol_index": str(args.protocol_index.resolve()),
        "protocol_index_sha256": sha(args.protocol_index),
        "processing_sources": source_hashes,
        "source_classical_search": state["contract"]["config"]["classical_search"],
        "expected_records": len(jobs),
        "nonfinal_debug_subset": args.allow_completed_subset,
        "device": args.device,
        "job_ids": [(direction, job["id"]) for direction, _, job, _ in jobs],
        "domain_shift": "dataset + device/reference + coordinate-frame/motion differences; no target calibration inferred",
        "normalization": "frozen SOURCE feature_stats/target_stats exclusively",
        "target_reference": "same family/variant/seed trained on target domain; descriptive reference, never a selected zero-shot winner",
        "postprocessing_runtime": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "precision": "FP32 no AMP/TF32; strict deterministic math attention",
            "cpu_threads": int(state["contract"]["config"].get("cpu_threads", 4)),
        },
    }
    if args.out.exists():
        if not args.resume:
            raise FileExistsError("Existing transfer output requires explicit --resume")
        if json.loads((args.out / "contract.json").read_text(encoding="utf-8")) != json_safe(
            contract
        ):
            raise ValueError("Transfer resume contract differs")
        if (args.out / "artifact_manifest.json").is_file():
            verify_record(args.out)
            print("SKIPS: verified complete immutable transfer output", flush=True)
            return 0
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        write_json_once(args.out / "contract.json", contract)
    search = _search_spec(contract["source_classical_search"])
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA unavailable; no CPU fallback")
    directions = {}
    records = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(int(state["contract"]["config"].get("cpu_threads", 4)))
    try:
        with _math_inference():
            for direction, family, job, reference in jobs:
                if direction not in directions:
                    frames, inputs = _load_direction(index, direction, project)
                    classical, cache_path = _classical_record(
                        args.out, direction, frames, search, inputs
                    )
                    directions[direction] = (frames, inputs, classical, cache_path)
                frames, inputs, classical, cache_path = directions[direction]
                label = f"{direction}__{family}__{job['phase']}__s{job['seed']}"
                destination = args.out / "records" / label
                if destination.exists():
                    verify_record(destination)
                    saved = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
                    if (
                        saved["checkpoint_sha256"] != job["checkpoint_sha256"]
                        or saved["target_fingerprint"] != inputs["physical_fingerprints"]["test"]
                    ):
                        raise ValueError("Completed transfer record identity changed")
                    records.append(saved)
                    print(f"SKIPS: verified completed transfer {label}", flush=True)
                    continue
                directory, config = _source_job(job, state, project)
                print(
                    f"READS: source-only checkpoint {directory / 'best.ckpt'} -> {direction}",
                    flush=True,
                )
                generator = VirtualIMUGenerator(directory / "best.ckpt", device=args.device)
                verify_run_provenance(directory, config, generator.config)
                original_stats = copy.deepcopy((generator.feature_stats, generator.target_stats))
                prediction, coverage = predict_complete_windows(
                    generator, frames["test"], batch_size=int(config["data"].get("batch_size", 128))
                )
                if (
                    generator.feature_columns != LOW
                    or generator.window_size not in {80, 160, 320}
                    or generator.stride != 40
                    or frame_fingerprint(prediction) != inputs["physical_fingerprints"]["test"]
                    or original_stats != (generator.feature_stats, generator.target_stats)
                ):
                    raise ValueError("Transfer changed source normalization or target support")
                methods = {
                    "model": prediction[VIRTUAL].to_numpy(),
                    **predict_source_classical(classical, prediction),
                }
                metrics = score_transfer(prediction, methods)
                result = {
                    "direction": direction,
                    "model_family": family,
                    "variant": job["phase"].removeprefix("final_"),
                    "seed": job["seed"],
                    "checkpoint": str(directory / "best.ckpt"),
                    "checkpoint_sha256": job["checkpoint_sha256"],
                    "source_config_sha256": job["run_config_sha256"],
                    "source_model_config": config["model"],
                    "source_feature_stats": original_stats[0],
                    "source_target_stats": original_stats[1],
                    "target_fingerprint": inputs["physical_fingerprints"]["test"],
                    "input_provenance": inputs,
                    "classical_cache": str(cache_path),
                    "classical_state_sha256": sha(cache_path / "state.json"),
                    "coverage": coverage,
                    "metrics": metrics,
                    "target_domain_reference": _reference(
                        reference, state, pipeline, prediction, project
                    ),
                    "nonfinal_debug_subset": args.allow_completed_subset,
                    "selection_uses_target_labels": False,
                }
                stage = _staging(args.out, label)
                write_json_once(stage / "metrics.json", result)
                prediction[
                    [*PHYSICAL, "original_sequence_id", *VIRTUAL, "prediction_overlap_count"]
                ].to_parquet(stage / "predictions.parquet", index=False)
                finish_record(stage, destination)
                records.append(json_safe(result))
                del generator
    finally:
        torch.set_num_threads(previous_threads)
    summary = _aggregate(records)
    if len(records) != contract["expected_records"]:
        raise RuntimeError("Transfer coverage incomplete")
    report = {
        "status": "debug_subset_complete" if args.allow_completed_subset else "complete",
        "created_utc": datetime.now(UTC).isoformat(),
        "contract_sha256": digest(contract),
        "records": len(records),
        "source_model_records": records,
        "seed_summary": summary,
        "scope": "zero-shot transfer with fixed source statistics; all declared models and variants retained",
        "interpretation": "Target-domain trained references are not zero-shot methods; optimizer seed SD is not a recording-level CI.",
    }
    report_path = args.out / "transfer_metrics.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            existing["contract_sha256"] != report["contract_sha256"]
            or existing["source_model_records"] != records
        ):
            raise ValueError("Completed transfer aggregate changed")
        print("SKIPS: unchanged completed transfer aggregate", flush=True)
    else:
        csv_path = args.out / "transfer_seed_summary.csv"
        csv_text = pd.DataFrame(summary).to_csv(index=False)
        if csv_path.exists():
            if csv_path.read_text(encoding="utf-8") != csv_text:
                raise ValueError("Existing transfer aggregate CSV differs")
        else:
            with csv_path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(csv_text)
        write_json_once(report_path, report)
    files = {
        path.relative_to(args.out).as_posix(): sha(path)
        for path in args.out.rglob("*")
        if path.is_file() and path.relative_to(args.out).parts[0] != "_partial"
    }
    write_json_once(
        args.out / "artifact_manifest.json", {"contract_sha256": digest(contract), "files": files}
    )
    print(f"WRITTEN: {len(records)} actual transfer evaluations; {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
