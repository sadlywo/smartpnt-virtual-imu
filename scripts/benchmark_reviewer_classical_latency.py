"""Time frozen reviewer classical fits on fixed, train-only physical CPU blocks.

Deploy this script together with evaluate_reviewer_transfer.py and
benchmark_forward_latency.py and the unchanged frozen src tree. Requires
threadpoolctl==3.6.0 (may live in an independent --target dependency folder).
No fitting, validation search, checkpoint loading or test-array decoding occurs.
FIR/ARX use preloaded same-segment history: this is steady-state block timing,
not sequence-start padding or zero-history window-reset timing.
"""

from __future__ import annotations

# ruff: noqa: E402 -- CLI supports both source checkout and isolated aux deployment.
import argparse
import copy
import hashlib
import inspect
import json
import os
import platform
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_forward_latency import latency_statistics
from evaluate_reviewer_transfer import (
    finish_record,
    sha,
    verify_pipeline,
    verify_record,
    write_json_once,
)
from virtual_imu.eval.baselines import _lag_features
from virtual_imu.eval.formal_evaluation import HIGH, IDENTITY, LOW
from virtual_imu.eval.formal_navigation import _window_summaries
from virtual_imu.eval.reviewer_evaluation import _digest, _frame_digest
from virtual_imu.io.training_split_manifest import load_training_split_manifest
from virtual_imu.training.gt_supervised_dataset import SMARTPNT_GT_IMU_COLUMNS
from virtual_imu.utils.paths import project_root_from_file

# Aux scripts can be stored separately from the immutable imported source tree.
PROJECT = project_root_from_file(inspect.getfile(_digest))

DEFAULTS = {
    "seed": 20261024,
    "warmup": 30,
    "iterations": 100,
    "repeats": 3,
    "batch_sizes": [1, 128],
    "paired_window_sizes": [80, 160, 320],
    "navigation_window_sizes": [80],
    "cpu_threads": 1,
}
SCOPE = (
    "CPU NumPy float64 algorithm forward with preloaded coefficients and physical LOW arrays; "
    "includes lag/delta construction, matrix/bias operations and navigation summary/standardization. "
    "Excludes file/dataframe I/O, window selection, history retrieval and model loading. "
    "FIR/ARX history is preloaded from the same contiguous TRAIN segment; steady-state block, "
    "not zero-history window reset. Identity returns an input view; constants allocate outputs. "
    "Neural receipts time FP32 network-only with normalization excluded; scopes/precisions differ. "
    "CPU and GPU measurements must be presented separately; no universal hardware ranking."
)


def array_digest(values):
    array = np.ascontiguousarray(values)
    return _digest(
        {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    )


def verify_saved(directory, *, exclude_partial=False):
    manifest = verify_record(directory)
    observed = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and path != directory / "artifact_manifest.json"
        and not (exclude_partial and "partial" in path.relative_to(directory).parts)
    }
    if observed != set(manifest["files"]):
        raise ValueError("Saved timing artifact checksum/inventory differs")
    return manifest


def _processing_sources(project, state):
    """Verify the complete trained src roster before reading any fitted state."""
    executed = {
        path.relative_to(project).as_posix(): sha(path)
        for path in (project / "src/virtual_imu").rglob("*.py")
    }
    matched = {}
    for original, expected in state.get("contract", {}).get("source_sha256", {}).items():
        normalized = str(original).replace("\\", "/")
        marker = "src/virtual_imu/"
        if normalized.startswith(marker):
            relative = normalized
        elif "/" + marker in normalized:
            relative = marker + normalized.split("/" + marker, 1)[1]
        else:
            continue
        if relative in matched and matched[relative] != expected:
            raise ValueError("Conflicting frozen training source identities")
        if executed.get(relative) != expected:
            raise ValueError(f"Executed source differs from frozen training source: {relative}")
        matched[relative] = expected
    if not matched or set(matched) != set(executed):
        raise ValueError(
            "Frozen training source roster differs from the complete executed src roster"
        )
    return {
        "source_root": str(project),
        "matched_trained_source_sha256": matched,
        "executed_source_sha256": executed,
        "standalone_script_sha256": sha(Path(__file__)),
    }


def _array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"Fitted {name} must have finite shape {shape}")
    return result


def method_specs(task, states):
    """Extract each effective fit once, preserving original method identifiers."""
    specs = []
    if task == "paired":
        for pair in states["pairs"]:
            fixed = pair["fixed"]
            base = {"task": task, "pair_identity": pair["identity"]}
            specs.extend(
                [
                    {**base, "method": "identity", "kind": "identity"},
                    {**base, "method": "bias_mean", "kind": "bias", "bias": fixed["bias"]},
                ]
            )
            for name in ("affine", "ridge_affine"):
                values = _array(fixed[name], (7, 6), name)
                specs.append(
                    {
                        **base,
                        "method": name,
                        "kind": "linear",
                        "weights": values[:-1].tolist(),
                        "bias": values[-1].tolist(),
                    }
                )
            system = fixed["system_id"]
            for name, prefix, delta in (
                ("learned_fir_causal", "fir", False),
                ("arx_linear", "arx", True),
            ):
                specs.append(
                    {
                        **base,
                        "method": name,
                        "kind": "tapped",
                        "taps": system[f"{prefix}_taps"],
                        "include_delta": delta,
                        "weights": system[f"{prefix}_weights"],
                        "bias": system[f"{prefix}_bias"],
                    }
                )
            for name in ("ridge_tuned", "fir_tuned", "arx_tuned"):
                fitted = pair[name]
                specs.append(
                    {
                        **base,
                        "method": name,
                        "kind": "tapped" if "taps" in fitted else "linear",
                        **{
                            key: fitted[key]
                            for key in ("weights", "bias", "taps", "include_delta")
                            if key in fitted
                        },
                    }
                )
    elif task == "navigation":
        base = {"task": task, "pair_identity": {}}
        specs.extend(
            [
                {**base, "method": "zero", "kind": "zero"},
                {
                    **base,
                    "method": "train_mean",
                    "kind": "constant",
                    "bias": states["fixed"]["train_mean_m"],
                },
            ]
        )
        for name, fitted in (
            ("ridge", states["fixed"]),
            ("ridge_tuned", states["ridge_tuned"]["state"]),
        ):
            coefficients = _array(fitted["ridge_coefficients"], (19, 3), name)
            specs.append(
                {
                    **base,
                    "method": name,
                    "kind": "navigation_ridge",
                    "weights": coefficients[:-1].tolist(),
                    "bias": coefficients[-1].tolist(),
                    "summary_mean": fitted["summary_mean"],
                    "summary_std": fitted["summary_std"],
                }
            )
    else:
        raise ValueError("Unknown classical task")
    for spec in specs:
        coefficient_inventory(spec)
    keys = [(tuple(sorted(spec["pair_identity"].items())), spec["method"]) for spec in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate pair/method state")
    return specs


def coefficient_inventory(spec):
    """Count fitted scalars, never call them torch trainable parameters."""
    kind = spec["kind"]
    output = 6 if spec["task"] == "paired" else 3
    features = 18 if kind == "navigation_ridge" else 6
    if kind == "tapped":
        taps = spec["taps"]
        if type(taps) is not int or taps < 1:
            raise ValueError("Lag taps must be positive integers")
        features *= taps + int(spec["include_delta"])
    weights = (
        _array(spec["weights"], (features, output), "weights")
        if "weights" in spec
        else np.empty((0,), dtype=np.float64)
    )
    bias = (
        _array(spec["bias"], (output,), "bias")
        if "bias" in spec
        else np.empty((0,), dtype=np.float64)
    )
    normalizer = np.empty((0,), dtype=np.float64)
    if kind == "navigation_ridge":
        mean = _array(spec["summary_mean"], (18,), "summary_mean")
        std = _array(spec["summary_std"], (18,), "summary_std")
        if np.any(std <= 0):
            raise ValueError("Summary standard deviations must be positive")
        normalizer = np.concatenate([mean, std])
    return {
        "coefficient_count": int(weights.size),
        "bias_count": int(bias.size),
        "fitted_scalar_count": int(weights.size + bias.size),
        "coefficient_bytes": int(weights.nbytes),
        "bias_bytes": int(bias.nbytes),
        "fitted_scalar_bytes": int(weights.nbytes + bias.nbytes),
        "normalizer_scalar_count": int(normalizer.size),
        "normalizer_scalar_bytes": int(normalizer.nbytes),
        "effective_state_bytes": int(weights.nbytes + bias.nbytes + normalizer.nbytes),
        "scalar_dtype": "float64",
        "not_torch_trainable_parameters": True,
    }


def compile_spec(spec):
    coefficient_inventory(spec)
    return {
        **spec,
        **{
            key: np.asarray(spec[key], dtype=np.float64)
            for key in ("weights", "bias", "summary_mean", "summary_std")
            if key in spec
        },
    }


def predict_block(spec, blocks, history=0):
    """Physical B x (history+T) x 6 -> B x T x 6 or B x 3."""
    values = np.asarray(blocks, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 6 or history < 0 or values.shape[1] <= history:
        raise ValueError("Physical input must have shape B x (history+T) x 6")
    low = values[:, history:]
    kind = spec["kind"]
    if kind == "identity":
        return low
    if kind == "bias":
        return low + np.asarray(spec["bias"])
    if kind == "linear":
        return low @ np.asarray(spec["weights"]) + np.asarray(spec["bias"])
    if kind == "tapped":
        needed = max(spec["taps"] - 1, int(spec["include_delta"]))
        if history < needed:
            raise ValueError("Steady-state FIR/ARX requires preloaded same-segment history")
        features = np.stack(
            [
                _lag_features(block, spec["taps"], spec["include_delta"])[history:]
                for block in values
            ]
        )
        return features @ np.asarray(spec["weights"]) + np.asarray(spec["bias"])
    if kind == "zero":
        return np.zeros((len(low), 3))
    if kind == "constant":
        return np.broadcast_to(spec["bias"], (len(low), 3)).copy()
    if kind == "navigation_ridge":
        summary = _window_summaries(low)
        normalized = (summary - np.asarray(spec["summary_mean"])) / np.asarray(spec["summary_std"])
        return normalized @ np.asarray(spec["weights"]) + np.asarray(spec["bias"])
    raise ValueError(f"Unknown classical kind {kind}")


def make_window_pool(frame, columns, window_size, history, size, seed):
    """Draw a fixed, unscored TRAIN block pool; history never crosses a segment."""
    if "split" in frame and set(frame.split.astype(str)) != {"train"}:
        raise ValueError("Timing input rows must be train only")
    if min(window_size, size) < 1 or history < 0:
        raise ValueError("Invalid timing block shape")
    groups, candidates = {}, []
    for name, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values("timestamp_s", kind="stable").reset_index(drop=True)
        values = group[columns].to_numpy(np.float64)
        if not np.isfinite(values).all() or not np.allclose(
            np.diff(group.timestamp_s), 0.025, atol=1e-6, rtol=1e-5
        ):
            raise ValueError("Timing input segments must be finite contiguous 40Hz")
        groups[str(name)] = (group, values)
        candidates.extend(
            (str(name), start)
            for start in range(0, len(group) - window_size + 1, 40)
            if start >= history
        )
    if not candidates:
        raise ValueError("No TRAIN windows with the required same-segment history")
    selection = np.random.default_rng(seed).choice(
        len(candidates), size=size, replace=len(candidates) < size
    )
    blocks, rows = [], []
    for index in selection:
        name, start = candidates[int(index)]
        group, values = groups[name]
        blocks.append(values[start - history : start + window_size])
        rows.append(
            {
                "sequence_id": name,
                "start_index": int(start),
                "history_start_index": int(start - history),
                "end_index": int(start + window_size - 1),
                "start_time_s": float(group.timestamp_s.iloc[start]),
                "history_start_time_s": float(group.timestamp_s.iloc[start - history]),
                "end_time_s": float(group.timestamp_s.iloc[start + window_size - 1]),
                "original_sequence_id": str(group.original_sequence_id.iloc[0])
                if "original_sequence_id" in group
                else name,
                "split": "train",
                "eligible_pool_windows": len(candidates),
                "sampling_with_replacement": len(candidates) < size,
            }
        )
    return np.stack(blocks), rows


def persist_record(
    out, identifier, contract, spec, values, history, *, warmup, iterations, repeats, resume
):
    destination = out / "records" / identifier
    if destination.exists():
        if not resume:
            raise FileExistsError(destination)
        verify_saved(destination)
        saved = json.loads((destination / "record.json").read_text(encoding="utf-8"))
        if saved["contract"] != contract:
            raise ValueError("Classical timing resume contract differs")
        print(f"SKIPS: verified completed timing {identifier}", flush=True)
        return saved
    staging = out / "partial" / (identifier + "__" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    write_json_once(staging / "contract.json", contract)
    fitted = compile_spec(spec)
    output = predict_block(fitted, values, history)
    expected = (
        (len(values), values.shape[1] - history, 6)
        if spec["task"] == "paired"
        else (len(values), 3)
    )
    if output.shape != expected or not np.isfinite(output).all():
        raise ValueError("Classical output shape or finiteness differs")
    for _ in range(warmup):
        predict_block(fitted, values, history)
    raw, loops = [], []
    for _ in range(repeats):
        timings = []
        for _ in range(iterations):
            start = time.perf_counter_ns()
            predict_block(fitted, values, history)
            timings.append((time.perf_counter_ns() - start) / 1e6)
        raw.append(timings)
        start = time.perf_counter_ns()
        for _ in range(iterations):
            predict_block(fitted, values, history)
        loops.append((time.perf_counter_ns() - start) / 1e9)
    record = {
        "contract": contract,
        "method": spec["method"],
        "task": spec["task"],
        "pair_identity": spec["pair_identity"],
        "device": "cpu",
        "precision": "NumPy float64",
        "input_shape_with_preloaded_history": list(values.shape),
        "history_samples": history,
        "window_size": values.shape[1] - history,
        "batch_size": len(values),
        "output_shape": list(output.shape),
        **coefficient_inventory(spec),
        "warmup": warmup,
        "iterations": iterations,
        "repeats": repeats,
        "wall": latency_statistics([v for part in raw for v in part]),
        "raw_wall_ms": raw,
        "raw_throughput_loop_seconds": loops,
        "loop_throughput_windows_per_second": len(values) * iterations * repeats / sum(loops),
        "output_sha256": array_digest(output),
        "scope": SCOPE,
    }
    write_json_once(staging / "record.json", record)
    finish_record(staging, destination)
    print(f"WRITES: timing {destination}", flush=True)
    return record


def _artifact_hash(job, path):
    expected = job.get("evaluation_artifact_sha256", {}).get(str(path.resolve()))
    if not expected or sha(path) != expected:
        raise ValueError(f"Frozen evaluation artifact checksum differs: {path}")
    return expected


def _training_files(payload, verified):
    reference = payload["contract"]["manifests"].get("split_manifest")
    if reference is None:
        raise ValueError("Frozen classical cache lacks its TRAIN split manifest")
    path = Path(reference["path"])
    path = path if path.is_absolute() else PROJECT / path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for entry in manifest["splits"]["train"]:
        source = Path(entry["path"])
        source = source if source.is_absolute() else PROJECT / source
        source = source.resolve()
        expected = entry.get("sha256")
        if not expected:
            raise ValueError("TRAIN record lacks an immutable file SHA")
        if source not in verified:
            if sha(source) != expected:
                raise ValueError(f"TRAIN file checksum differs: {source}")
            verified[source] = expected
            print(f"READS: verifies TRAIN file bytes only {source}", flush=True)
        elif verified[source] != expected:
            raise ValueError("Conflicting TRAIN source file fingerprints")
        records.append(
            {
                "path": str(source),
                "sha256": expected,
                "sequence_id": entry["sequence_id"],
                "original_sequence_id": entry.get("original_sequence_id"),
                "split": "train",
            }
        )
    if not records:
        raise ValueError("Frozen classical state has no declared TRAIN records")
    return records


def collect_sources(state, pipeline):
    """Gate all finals, then hash only metadata/state; never read test arrays."""
    finals = verify_pipeline(state, pipeline, allow_subset=False)
    _processing_sources(PROJECT, state)
    unique, mapping = {}, []
    declarations = state["contract"]["config"]["protocols"]
    expected = {row["id"]: row["task"] for row in declarations}
    if (
        len(expected) != 9
        or list(expected.values()).count("paired") != 6
        or list(expected.values()).count("navigation") != 3
    ):
        raise ValueError("Classical timing requires the declared nine-protocol matrix")
    cached, verified_training_files, training_records = {}, {}, {}
    for job in finals:
        directory = Path(job["run_dir"]).resolve()
        if not directory.is_relative_to(pipeline / "experiments"):
            raise ValueError("Final run is outside the exact pipeline experiments prefix")
        config_path, states_path = (
            directory / "config.yaml",
            directory / "evaluation/baseline_train_state.json",
        )
        if sha(config_path) != job["run_config_sha256"]:
            raise ValueError("Final config checksum differs")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        protocol = config["reviewer_protocol"]["protocol_id"]
        task = expected.get(protocol)
        if task is None or job["cell"].split("/", 1)[0] != protocol:
            raise ValueError("Protocol/config identity differs")
        metadata = config["reviewer_protocol"]
        if (
            metadata["phase"] != job["phase"]
            or metadata["variant"] != job["phase"].removeprefix("final_")
            or metadata["model_family"] != job["cell"].split("/", 1)[1]
            or config["seed"] != job["seed"]
            or config["model"]["in_channels"] != 6
            or config["model"]["out_channels"] != (6 if task == "paired" else 3)
            or config["data"]["window_size"] not in ([80, 160, 320] if task == "paired" else [80])
        ):
            raise ValueError("Frozen final model/task/phase/shape metadata differs")
        summary = pipeline / "evaluations" / f"{job['id']}.json"
        if sha(summary) != job["evaluation_sha256"]:
            raise ValueError("Final evaluation summary checksum differs")
        metrics_path = directory / "evaluation/metrics.json"
        _artifact_hash(job, metrics_path)
        state_file_sha = _artifact_hash(job, states_path)
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        cache_ref = report["classical_cache"]
        path = Path(cache_ref["path"]).resolve()
        if not path.is_relative_to(pipeline / "classical" / protocol):
            raise ValueError("Classical cache is outside the protocol prefix")
        if _artifact_hash(job, path) != cache_ref["sha256"]:
            raise ValueError("Classical cache reference checksum differs")
        if path not in cached:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload["state_sha256"] != _digest(payload["states"])
                or _digest(payload["contract"]) != cache_ref["contract_sha256"]
            ):
                raise ValueError("Classical cache state/contract digest differs")
            if (
                payload["contract"]["protocol_id"] != protocol
                or payload["contract"]["task"] != task
            ):
                raise ValueError("Cache task/protocol identity differs")
            if (
                payload["states"].get("fit_split") != "train"
                or payload["states"].get("selection_split") != "validation"
            ):
                raise ValueError("Classical state was not train-fit/validation-selected")
            for implementation, expected_sha in payload["contract"]["implementation"].items():
                if sha(Path(implementation)) != expected_sha:
                    raise ValueError("Frozen baseline implementation changed")
            for name, reference in payload["contract"]["manifests"].items():
                source = Path(reference["path"])
                source = source if source.is_absolute() else PROJECT / source
                if (
                    sha(source) != reference["sha256"]
                    or config["data"].get(name) != reference["path"]
                ):
                    raise ValueError("Source manifest checksum/config identity differs")
            cached[path] = payload
            training_records[path] = _training_files(payload, verified_training_files)
            print(f"READS: frozen classical state/cache {path}", flush=True)
        payload = cached[path]
        if any(
            config["data"].get(name) != reference["path"]
            for name, reference in payload["contract"]["manifests"].items()
        ):
            raise ValueError("Repeated state has different source manifest identity")
        saved = json.loads(states_path.read_text(encoding="utf-8"))
        if saved != payload["states"]:
            raise ValueError("Final baseline state differs from its frozen cache")
        key = (
            protocol
            + "__"
            + _digest({"contract": payload["contract"], "state_sha256": payload["state_sha256"]})[
                :20
            ]
        )
        unique.setdefault(
            key,
            {
                "id": key,
                "protocol_id": protocol,
                "task": task,
                "config": config,
                "states": saved,
                "cache": payload,
                "cache_path": str(path),
                "cache_sha256": sha(path),
                "training_inputs": training_records[path],
            },
        )
        mapping.append(
            {
                "job_id": job["id"],
                "cell": job["cell"],
                "phase": job["phase"],
                "seed": job["seed"],
                "source_id": key,
                "protocol_id": protocol,
                "task": task,
                "run_config_sha256": job["run_config_sha256"],
                "baseline_state_path": str(states_path),
                "baseline_state_file_sha256": state_file_sha,
                "cache_path": str(path),
                "state_sha256": payload["state_sha256"],
                "contract_sha256": _digest(payload["contract"]),
            }
        )
    if set(row["protocol_id"] for row in unique.values()) != set(expected):
        raise ValueError("Incomplete classical protocol coverage")
    print(
        f"SKIPS: duplicate fits after state/train-validation row checks; {len(mapping)} final mappings -> {len(unique)} independent caches",
        flush=True,
    )
    return list(unique.values()), mapping


def load_train(source):
    config, task = source["config"], source["task"]
    contract = source["cache"]["contract"]["train_validation"]
    data = copy.deepcopy(config["data"])
    if task == "paired":
        bundle = load_training_split_manifest(
            Path(data["split_manifest"]), Path(data["strict_manifest"]), split_names=["train"]
        )
        pieces = [
            group.sort_values("timestamp_s").reset_index(drop=True)
            for _, group in bundle.frames["train"].groupby("sequence_id", sort=False)
            if len(group) >= 320
        ]
        if not pieces:
            raise ValueError("No eligible paired TRAIN input")
        frame = pd.concat(pieces, ignore_index=True)
        if (
            len(frame) != contract["train_rows"]
            or _frame_digest(frame, [*IDENTITY, "sequence_id", "timestamp_s", *LOW, *HIGH])
            != contract["train_sha256"]
        ):
            raise ValueError(
                "TRAIN physical row fingerprint differs from the fitted classical cache"
            )
        return frame, LOW
    pieces = []
    for record in source["training_inputs"]:
        print(f"READS: navigation TRAIN LOW columns {record['path']}", flush=True)
        piece = pd.read_parquet(
            record["path"],
            columns=[
                *SMARTPNT_GT_IMU_COLUMNS,
                "sequence_id",
                "original_sequence_id",
                "timestamp_s",
            ],
        )
        if set(piece.sequence_id.astype(str)) != {record["sequence_id"]} or set(
            piece.original_sequence_id.astype(str)
        ) != {record["original_sequence_id"]}:
            raise ValueError("Navigation TRAIN row identity differs from its frozen record")
        piece["split"] = "train"
        pieces.append(piece)
    frame = pd.concat(pieces, ignore_index=True)
    if set(frame.original_sequence_id.astype(str)) != set(data["train_sequences"]):
        raise ValueError("Navigation timing TRAIN role differs")
    digest, count = hashlib.sha256(), 0
    for _, group in frame.groupby("sequence_id", sort=False):
        values = group.sort_values("timestamp_s")[SMARTPNT_GT_IMU_COLUMNS].to_numpy(np.float32)
        for start in range(0, len(values) - 80 + 1, 40):
            digest.update(values[start : start + 80].tobytes())
            count += 1
    if (
        digest.hexdigest() != contract["train"]["features_sha256"]
        or count != contract["train"]["windows"]
    ):
        raise ValueError("Navigation TRAIN feature fingerprint differs from fitted cache")
    # _physical_windows freezes LOW values as float32; preserve that rounding
    # before the original float64 classical summary arithmetic.
    frame[SMARTPNT_GT_IMU_COLUMNS] = frame[SMARTPNT_GT_IMU_COLUMNS].astype(np.float32)
    return frame, SMARTPNT_GT_IMU_COLUMNS


def _thread_environment():
    try:
        import threadpoolctl
    except ImportError as exc:
        raise RuntimeError(
            "Install pinned threadpoolctl==3.6.0 in an independent dependency target; NumPy thread count must be verified"
        ) from exc
    if threadpoolctl.__version__ != "3.6.0":
        raise ValueError("This timing contract pins threadpoolctl==3.6.0")
    return threadpoolctl


def _csvs(out, records, mapping):
    rows, raw = [], []
    for record in records:
        contract = record["contract"]
        base = {
            "protocol_id": contract["protocol_id"],
            "source_id": contract["source_id"],
            "method": record["method"],
            "pair_identity": json.dumps(record["pair_identity"], sort_keys=True),
            "task": record["task"],
            "window_size": record["window_size"],
            "history_samples": record["history_samples"],
            "batch_size": record["batch_size"],
            "state_sha256": contract["state_sha256"],
            "input_sha256": contract["input_sha256"],
        }
        rows.append(
            {
                **base,
                **{k: v for k, v in record.items() if k.endswith("_count") or k.endswith("_bytes")},
                **record["wall"],
                "loop_throughput_windows_per_second": record["loop_throughput_windows_per_second"],
                "device": "cpu",
                "precision": record["precision"],
            }
        )
        for repeat, values in enumerate(record["raw_wall_ms"]):
            raw.extend(
                {**base, "repeat": repeat, "iteration": index, "wall_ms": value}
                for index, value in enumerate(values)
            )
    for name, data in (
        ("classical_latency.csv", rows),
        ("classical_latency_raw.csv", raw),
        ("final_run_mapping.csv", mapping),
    ):
        path = out / name
        serialized = pd.DataFrame(data).to_csv(index=False, lineterminator="\n")
        if path.exists():
            if path.read_text(encoding="utf-8") != serialized:
                raise ValueError("Existing completed summary content differs")
            continue
        with path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(serialized)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=SCRIPT_ROOT / "configs/reviewer_classical_latency.yaml"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    pipeline, out = args.pipeline_root.resolve(), args.out.resolve()
    print(f"READS: completed reviewer pipeline metadata {pipeline / 'pipeline.json'}", flush=True)
    state = json.loads((pipeline / "pipeline.json").read_text(encoding="utf-8"))
    # Gate precedes dependency setup, output creation and any train/test payload access.
    sources, mapping = collect_sources(state, pipeline)
    timing = {**DEFAULTS, **yaml.safe_load(args.config.read_text(encoding="utf-8"))}
    if timing != DEFAULTS:
        raise ValueError(
            "Production timing must use the predeclared seed/30warmup/100x3/B1,B128/80,160,320/single-thread contract"
        )
    threadpoolctl = _thread_environment()
    with threadpoolctl.threadpool_limits(limits=1):
        pools = threadpoolctl.threadpool_info()
        if not pools or any(pool["num_threads"] != 1 for pool in pools):
            raise ValueError("All loaded native thread pools must report one CPU thread")
        implementations = {
            str(Path(__file__).resolve()): sha(Path(__file__)),
            **{
                str(Path(inspect.getfile(fn)).resolve()): sha(Path(inspect.getfile(fn)))
                for fn in (
                    verify_pipeline,
                    latency_statistics,
                    _lag_features,
                    _window_summaries,
                    load_training_split_manifest,
                )
            },
        }
        environment = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "threadpoolctl_version": threadpoolctl.__version__,
            "threadpoolctl_origin": threadpoolctl.__file__,
            "threadpoolctl_source_sha256": sha(Path(threadpoolctl.__file__)),
            "native_thread_pools": pools,
            "logical_cpu_count": os.cpu_count(),
        }
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            environment["cpu_model_name"] = next(
                (
                    line.split(":", 1)[1].strip()
                    for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")
                ),
                "unknown",
            )
        contract = {
            "schema_version": 1,
            "pipeline": str(pipeline),
            "pipeline_sha256": sha(pipeline / "pipeline.json"),
            "config_sha256": sha(args.config),
            "timing": timing,
            "scope": SCOPE,
            "implementations": implementations,
            "processing_sources": _processing_sources(PROJECT, state),
            "training_input_files": {source["id"]: source["training_inputs"] for source in sources},
            "environment": environment,
            "final_mapping": mapping,
        }
        if out.exists() and any(out.iterdir()):
            if not args.resume:
                raise FileExistsError("Nonempty output requires --resume")
            if json.loads((out / "contract.json").read_text(encoding="utf-8")) != contract:
                raise ValueError("Production resume contract/source/environment differs")
        else:
            out.mkdir(parents=True, exist_ok=True)
            write_json_once(out / "contract.json", contract)
            (out / "config_snapshot.yaml").write_bytes(args.config.read_bytes())
            (out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
        if (out / "artifact_manifest.json").exists():
            verify_saved(out, exclude_partial=True)
            print(
                "SKIPS: complete classical timing and every saved artifact hash verified",
                flush=True,
            )
            return 0
        print(
            "SKIPS: test payloads/labels, refitting, checkpoint inference, GPU timing; partial work is retained",
            flush=True,
        )
        records, coverage = [], []
        timing_contract_sha256 = _digest(contract)
        for source in sources:
            frame, channels = load_train(source)
            specs = method_specs(source["task"], source["states"])
            history = max(
                (
                    max(spec["taps"] - 1, int(spec["include_delta"]))
                    for spec in specs
                    if spec["kind"] == "tapped"
                ),
                default=0,
            )
            contexts = (
                timing["paired_window_sizes"]
                if source["task"] == "paired"
                else timing["navigation_window_sizes"]
            )
            for pair_items in dict.fromkeys(
                tuple(sorted(spec["pair_identity"].items())) for spec in specs
            ):
                identity = dict(pair_items)
                selected = frame
                for key, value in identity.items():
                    selected = selected[selected[key].astype(str).eq(str(value))]
                if selected.empty:
                    raise ValueError("Frozen pair state has no matching TRAIN rows")
                pair_key = _digest(identity)[:12]
                for context in contexts:
                    values, metadata = make_window_pool(
                        selected, channels, context, history, 128, timing["seed"]
                    )
                    pool_id = f"{source['id']}__{pair_key}__t{context}"
                    pool_destination = out / "input_pools" / pool_id
                    pool_contract = {
                        "source_id": source["id"],
                        "pair_identity": identity,
                        "window_size": context,
                        "history_samples": history,
                        "array_sha256": array_digest(values),
                        "rows": metadata,
                    }
                    if pool_destination.exists():
                        verify_saved(pool_destination)
                        if (
                            json.loads((pool_destination / "rows.json").read_text(encoding="utf-8"))
                            != pool_contract
                        ):
                            raise ValueError("Fixed TRAIN input pool changed")
                    else:
                        stage = out / "partial" / (pool_id + "__" + uuid.uuid4().hex)
                        stage.mkdir(parents=True)
                        write_json_once(stage / "rows.json", pool_contract)
                        np.save(
                            stage / "physical_low_history_blocks.npy", values, allow_pickle=False
                        )
                        finish_record(stage, pool_destination)
                    for spec in (
                        s for s in specs if tuple(sorted(s["pair_identity"].items())) == pair_items
                    ):
                        for batch in timing["batch_sizes"]:
                            inputs = values[:batch]
                            record_contract = {
                                "protocol_id": source["protocol_id"],
                                "source_id": source["id"],
                                "source_cache_sha256": source["cache_sha256"],
                                "state_sha256": _digest(spec),
                                "input_sha256": array_digest(inputs),
                                "pool_id": pool_id,
                                "timing_contract_sha256": timing_contract_sha256,
                            }
                            identifier = f"{pool_id}__{spec['method']}__b{batch}"
                            record = persist_record(
                                out,
                                identifier,
                                record_contract,
                                spec,
                                inputs,
                                history,
                                warmup=timing["warmup"],
                                iterations=timing["iterations"],
                                repeats=timing["repeats"],
                                resume=args.resume,
                            )
                            records.append(record)
                            coverage.append(
                                {
                                    "record_id": identifier,
                                    **record_contract,
                                    "task": source["task"],
                                    "pair_identity": identity,
                                    "method": spec["method"],
                                    "window_size": context,
                                    "batch_size": batch,
                                }
                            )
        _csvs(out, records, mapping)
        report = {
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "contract_sha256": _digest(contract),
            "scope": SCOPE,
            "environment": environment,
            "independent_cache_count": len(sources),
            "final_run_mappings": len(mapping),
            "coverage": coverage,
            "records": records,
            "statistics_note": "P50/P95 are call-distribution quantiles; no independent-seed CI; duplicate neural slots do not create classical replicates.",
        }
        report_path = out / "latency.json"
        if report_path.exists():
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            report["created_utc"] = previous.get("created_utc")
            if previous != report:
                raise ValueError("Existing partial summary differs from verified records")
        else:
            write_json_once(report_path, report)
        files = {
            path.relative_to(out).as_posix(): sha(path)
            for path in out.rglob("*")
            if path.is_file() and "partial" not in path.relative_to(out).parts
        }
        write_json_once(out / "artifact_manifest.json", {"files": files})
        print(
            f"WRITES: complete classical CPU timing {out}: {len(records)} distinct records; {len(mapping)} source mappings",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
