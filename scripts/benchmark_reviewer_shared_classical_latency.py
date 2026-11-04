"""Independent, train-only NumPy float64 shared-state and bank CPU measurements.

Only run after the original workflows and all shared evaluation stages finish.
The timed kernels include feature construction. Physical history is resident,
is never borrowed from another stream, and is never reset at a window boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from reviewer_shared_classical import (
    IDENTITY,
    LOW,
    METHODS,
    PROFILE,
    PROTOCOLS,
    SCOPE,
    digest,
    finish_stage,
    frame_sha,
    normalize_bank_pair,
    predict_state,
    read_json,
    sha,
    validate_streams,
    verify_artifact_manifest,
    verify_state,
    write_once,
)
from virtual_imu.eval.baselines import _lag_features

TIMING_SCOPE = (
    "CPU NumPy float64 algorithm forward including lag/delta feature construction and matrix/bias math; "
    "preloaded same-stream actual past, no per-window history reset. Excludes I/O, dataframe conversion, "
    "fit/selection, model loading, history retrieval/initialization and input scheduling. "
    "Whole bank workloads execute every pair in the frozen scheduler; they are not single-device costs. "
    "Original neural FP32 network-only timings have a different normalization/precision scope. "
    "No universal CPU/GPU ranking or 40Hz feasibility claim."
)
DEFAULTS = {
    "schema_version": 1,
    "analysis_profile": PROFILE,
    "seed": 20261024,
    "warmup": 30,
    "iterations": 100,
    "repeats": 3,
    "batch_sizes": [1, 128],
    "window_sizes": [80, 160, 320],
    "cpu_threads": 1,
    "threadpoolctl_version": "3.6.0",
    "dtype": "float64",
    "input_role": "train",
    "history": "same_stream_preloaded",
    "scope": "feature_construction_and_algorithm_forward",
}


def assert_no_competing_experiments():
    """Require actual process-level quiet on the Linux server, outside timing."""
    proc = Path("/proc")
    if not proc.is_dir():
        raise RuntimeError("Formal CPU timing requires Linux /proc workload verification")
    names = {
        "train_gt_supervised.py",
        "run_reviewer_hpo.py",
        "run_benchmark_training.py",
        "run_formal_training.py",
        "run_alimu_leave_sensor_matrix.py",
        "run_penn_loso_matrix.py",
        "benchmark_reviewer_latency.py",
        "benchmark_reviewer_classical_latency.py",
        "benchmark_reviewer_shared_classical_latency.py",
        "benchmark_forward_latency.py",
        "evaluate_reviewer_transfer.py",
        "evaluate_reviewer_shared_classical.py",
        "evaluate_reviewer_shared_downstream.py",
        "evaluate_reviewer_downstream.py",
    }
    active = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            tokens = (entry / "cmdline").read_bytes().decode("utf-8", errors="replace").split("\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        matched = sorted({Path(token).name for token in tokens if Path(token).name in names})
        if matched:
            active.append({"pid": int(entry.name), "script_names": matched})
    if active:
        raise RuntimeError(
            f"Competing experiment training/inference/timing process detected: {active}"
        )
    return {
        "status": "passed",
        "scope": "Linux process argv script names; no environment or full argv recorded",
        "checked_pid": os.getpid(),
    }


def array_sha(values):
    array = np.ascontiguousarray(values, dtype=np.float64)
    return digest(
        {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "bytes_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    )


def coefficient_inventory(state):
    """Count actual finite arrays, with the intercept included in coefficient_count."""
    verify_state(state)
    weights = np.asarray(state["weights"], dtype=np.float64)
    bias = np.asarray(state["intercept"], dtype=np.float64)
    spec = state["feature_spec"]
    kind = state["state_kind"]
    expected = (
        0
        if kind == "identity"
        else 6
        if kind == "bias"
        else 36 * ((spec["taps"] or 1) + int(spec["include_delta"])) + 6
    )
    count, size = weights.size + bias.size, weights.nbytes + bias.nbytes
    if (
        count != expected
        or count != state["coefficient_count"]
        or size != state["coefficient_bytes"]
    ):
        raise ValueError("Actual array coefficients differ from normalized state accounting")
    if state["normalization"] != {"mode": "none_raw_si", "coefficient_count": 0}:
        raise ValueError("Unexpected fitted normalization state")
    return {
        "weight_count": int(weights.size),
        "intercept_count": int(bias.size),
        "coefficient_count": int(count),
        "coefficient_bytes": int(size),
        "normalizer_count": 0,
        "normalizer_bytes": 0,
        "scalar_dtype": "float64",
        "coefficient_interpretation": "stored_learned_float64_scalars_not_statistical_DoF",
        "not_torch_trainable_parameters": True,
    }


def compile_state(state):
    coefficient_inventory(state)
    weights, bias = (
        np.asarray(state["weights"], dtype=np.float64),
        np.asarray(state["intercept"], dtype=np.float64),
    )
    return {
        "state": state,
        "weights": weights,
        "bias": bias,
        "design_coefficients": np.vstack([weights, bias])
        if state["state_kind"] == "linear"
        else None,
    }


def predict_blocks(compiled, blocks, history=0):
    """B x (resident history + W) x 6 -> B x W x 6, using source arithmetic."""
    values = np.asarray(blocks, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 6 or history < 0 or history >= values.shape[1]:
        raise ValueError("Physical timing block must have shape B x (history+W) x 6")
    state, low = compiled["state"], values[:, history:]
    kind = state["state_kind"]
    if kind == "identity":
        return low
    if kind == "bias":
        return low + compiled["bias"]
    spec = state["feature_spec"]
    features = (
        np.stack(
            [
                _lag_features(block, spec["taps"], spec["include_delta"])[history:]
                for block in values
            ]
        )
        if spec["taps"]
        else low
    )
    # Preserve the original fixed affine/ridge augmented-design operation rather
    # than silently replacing it with a potentially differently rounded W+b.
    if state["arithmetic"] == "design_matmul":
        flat = features.reshape(-1, features.shape[-1])
        result = np.column_stack([flat, np.ones(len(flat))]) @ compiled["design_coefficients"]
        return result.reshape(*features.shape[:-1], 6)
    if state["arithmetic"] != "weights_bias":
        raise ValueError("Unrecognized source arithmetic")
    return features @ compiled["weights"] + compiled["bias"]


def make_window_pool(frame, window_size, history, size, seed):
    """Freeze one pair's unscored TRAIN scheduler and physical rows before timing."""
    validate_streams(frame, require_target=False)
    if "split" not in frame or set(frame.split.astype(str)) != {"train"}:
        raise ValueError("Timing input must explicitly be TRAIN only")
    if len(frame[IDENTITY].drop_duplicates()) != 1 or min(window_size, size) < 1 or history < 0:
        raise ValueError("Each timing pool must contain one physical pair with a valid shape")
    groups, candidates = {}, []
    for name, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values("timestamp_s", kind="stable").reset_index(drop=True)
        if not np.allclose(np.diff(group.timestamp_s), 0.025, atol=1e-6, rtol=1e-5):
            raise ValueError("Timing streams must be contiguous 40Hz")
        groups[str(name)] = group
        candidates.extend(
            (str(name), start)
            for start in range(0, len(group) - window_size + 1, 40)
            if start >= history
        )
    if not candidates:
        raise ValueError("No TRAIN window has enough real same-stream past")
    selected = np.random.default_rng(seed).choice(
        len(candidates), size, replace=len(candidates) < size
    )
    blocks, rows = [], []
    for number in selected:
        sequence, start = candidates[int(number)]
        group = groups[sequence]
        physical = group.iloc[start - history : start + window_size]
        values = physical[LOW].to_numpy(np.float64)
        blocks.append(values)
        columns = [*IDENTITY, "sequence_id", "original_sequence_id", "timestamp_s", *LOW]
        rows.append(
            {
                "sequence_id": sequence,
                "original_sequence_id": str(group.original_sequence_id.iloc[0]),
                "start_index": int(start),
                "history_start_index": int(start - history),
                "end_index": int(start + window_size - 1),
                "split": "train",
                "start_time_s": float(group.timestamp_s.iloc[start]),
                "history_start_time_s": float(group.timestamp_s.iloc[start - history]),
                "end_time_s": float(group.timestamp_s.iloc[start + window_size - 1]),
                "physical_row_sha256": frame_sha(physical, columns),
                "input_sha256": array_sha(values),
                "eligible_pool_windows": len(candidates),
                "sampling_with_replacement": len(candidates) < size,
            }
        )
    return np.stack(blocks), rows


def parameter_tables(records):
    """Deduplicate repeated references, keeping distinct pair deployments additive."""
    unique, references, artifacts = {}, {}, {}
    for record in records:
        state = record["state"]
        pair = json.dumps(record.get("pair_identity", {}), sort_keys=True, separators=(",", ":"))
        group = (
            record["protocol"],
            record["information_scope"],
            record["canonical_method"],
            record["variant"],
        )
        key = (*group, pair, state["numeric_sha256"])
        references[group] = references.get(group, 0) + 1
        if record.get("state_path"):
            artifacts.setdefault(group, {})[record["state_path"]] = (
                record.get("serialized_state_bytes") or 0
            )
        if key not in unique:
            unique[key] = {
                **{
                    name: record[name]
                    for name in ("protocol", "information_scope", "canonical_method", "variant")
                },
                "pair_identity_json": pair,
                "state_content_sha256": digest(state),
                "numeric_sha256": state["numeric_sha256"],
                **coefficient_inventory(state),
                "state_path": record.get("state_path"),
                "state_file_sha256": record.get("state_file_sha256"),
                "serialized_state_bytes": record.get("serialized_state_bytes"),
                "reference_count": 0,
            }
        unique[key]["reference_count"] += 1
    inventory, totals = list(unique.values()), []
    for group, count in references.items():
        rows = [
            row
            for row in inventory
            if tuple(
                row[name]
                for name in ("protocol", "information_scope", "canonical_method", "variant")
            )
            == group
        ]
        numeric = {row["numeric_sha256"]: row for row in rows}
        totals.append(
            {
                **dict(
                    zip(
                        ("protocol", "information_scope", "canonical_method", "variant"),
                        group,
                        strict=True,
                    )
                ),
                "pair_count": len({row["pair_identity_json"] for row in rows}),
                "effective_pair_states": len(rows),
                "reference_count": count,
                "bank_coefficient_count": sum(row["coefficient_count"] for row in rows),
                "bank_coefficient_bytes": sum(row["coefficient_bytes"] for row in rows),
                "normalizer_count": sum(row["normalizer_count"] for row in rows),
                "normalizer_bytes": sum(row["normalizer_bytes"] for row in rows),
                "numeric_unique_states": len(numeric),
                "numeric_unique_coefficient_count": sum(
                    row["coefficient_count"] for row in numeric.values()
                ),
                "numeric_unique_coefficient_bytes": sum(
                    row["coefficient_bytes"] for row in numeric.values()
                ),
                "requires_known_pair_routing": group[1] == "known_device_bank",
                "all_artifact_serialized_bytes": sum(artifacts.get(group, {}).values()),
                "all_artifact_file_count": len(artifacts.get(group, {})),
                "artifact_byte_scope": "unique_whole_state_files_may_include_other_methods_not_deployed_coefficient_bytes",
            }
        )
    return inventory, totals


def time_workload(states, pools, *, history, warmup, iterations, repeats, predictor=predict_blocks):
    """Actually time the complete ordered P-pair loop and a separate throughput loop."""
    if not states or len(states) != len(pools) or min(iterations, repeats) < 1 or warmup < 0:
        raise ValueError("Invalid complete workload")

    def call():
        return [predictor(state, pool, history) for state, pool in zip(states, pools, strict=True)]

    for _ in range(warmup):
        call()
    raw, loops = [], []
    for _ in range(repeats):
        for _ in range(iterations):
            start = time.perf_counter_ns()
            call()
            raw.append((time.perf_counter_ns() - start) / 1e6)
        start = time.perf_counter_ns()
        for _ in range(iterations):
            call()
        loops.append((time.perf_counter_ns() - start) / 1e9)
    windows = sum(len(pool) for pool in pools)
    return {
        "pair_count": len(pools),
        "windows_per_call": windows,
        "p50_ms": float(np.percentile(raw, 50)),
        "p95_ms": float(np.percentile(raw, 95)),
        "call_distribution_sd_ms": float(np.std(raw, ddof=1)) if len(raw) > 1 else None,
        "raw_wall_ms": raw,
        "raw_throughput_loop_seconds": loops,
        "loop_throughput_windows_per_second": windows * iterations * repeats / sum(loops),
        "warmup": warmup,
        "iterations": iterations,
        "repeats": repeats,
        "throughput_scope": "separate_complete_workload_loop",
    }


@contextmanager
def single_thread_runtime():
    import threadpoolctl

    if threadpoolctl.__version__ != "3.6.0":
        raise ValueError("CPU timing requires exact threadpoolctl 3.6.0")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(variable) != "1":
            raise ValueError(f"CPU timing requires {variable}=1 before process startup")
    with threadpoolctl.threadpool_limits(limits=1):
        # Force the BLAS runtime to load before inspecting the actual threadpool.
        np.ones((6, 6), dtype=np.float64) @ np.ones((6, 6), dtype=np.float64)
        pools = threadpoolctl.threadpool_info()
        if not pools or any(pool.get("num_threads") != 1 for pool in pools):
            raise ValueError("Actual native threadpool is absent or not single-threaded")
        yield {
            "threadpoolctl_version": threadpoolctl.__version__,
            "threadpools": pools,
            "numpy_version": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        }
        if any(pool.get("num_threads") != 1 for pool in threadpoolctl.threadpool_info()):
            raise ValueError("Native threadpool changed during CPU measurement")


def _load_bank_states(core, pipeline, project):
    from evaluate_reviewer_shared_classical import original_helpers

    helper, unique, references = original_helpers(), {}, []
    for job in core["jobs"].values():
        protocol = job["cell"].split("/", 1)[0]
        if not job["phase"].startswith("final_") or protocol not in PROTOCOLS:
            continue
        path = (
            helper.recorded_path(job["run_dir"], pipeline, project)
            / "evaluation/baseline_train_state.json"
        )
        expected = {
            str(helper.recorded_path(name, pipeline, project).resolve()): checksum
            for name, checksum in job.get("evaluation_artifact_sha256", {}).items()
        }
        if expected.get(str(path.resolve())) != sha(path):
            raise ValueError("Actual bank state is not bound to its original final artifact hash")
        saved = read_json(path)
        if protocol in unique and saved != unique[protocol]["saved"]:
            raise ValueError("Repeated bank final mappings have different actual fitted states")
        unique.setdefault(
            protocol,
            {
                "saved": saved,
                "path": str(path),
                "sha256": sha(path),
                "serialized_bytes": path.stat().st_size,
            },
        )
        references.append(
            {
                "protocol": protocol,
                "job_id": job["id"],
                "state_path": str(path),
                "sha256": sha(path),
            }
        )
    if set(unique) != set(PROTOCOLS) or len(references) != 360:
        raise ValueError("Bank state source must cover all 360 paired final mappings")
    return unique, references


def verify_kernel(state, frame):
    """Independent full-frame predictor proves stream-start and all pool interior math."""
    compiled = compile_state(state)
    results = []
    for sequence, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values("timestamp_s", kind="stable").reset_index(drop=True)
        expected = predict_state(state, group)
        values = group[LOW].to_numpy(np.float64)
        width = min(80, len(group))
        np.testing.assert_allclose(
            predict_blocks(compiled, values[None, :width], 0)[0],
            expected[:width],
            rtol=1e-12,
            atol=1e-12,
        )
        history = max(
            (state["feature_spec"]["taps"] or 1) - 1, int(state["feature_spec"]["include_delta"])
        )
        if len(values) < history + width:
            raise ValueError("Insufficient continuous TRAIN for kernel proof")
        start = max(history, len(values) // 2)
        end = min(start + width, len(values))
        measured = predict_blocks(compiled, values[None, start - history : end], history)[0]
        np.testing.assert_allclose(measured, expected[start:end], rtol=1e-12, atol=1e-12)
        results.append(
            {
                "sequence_id": str(sequence),
                "start_checked": True,
                "interior_checked": True,
                "history_samples": history,
                "max_abs_error": float(np.max(np.abs(measured - expected[start:end]))),
            }
        )
    return results


def run(args):
    from evaluate_reviewer_shared_classical import load_roles, metadata_gate, verify_selection_set

    project, pipeline, out = (
        args.project_root.resolve(),
        args.pipeline_root.resolve(),
        args.out.resolve(),
    )
    print(
        f"READS: frozen selections, actual bank arrays and TRAIN streams; WRITES: {out}/cpu_latency; SKIPS: fit, TEST decoding, neural inference",
        flush=True,
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config != DEFAULTS:
        raise ValueError(
            "Shared CPU budget/seed/thread/history contract must match the frozen full configuration"
        )
    shared_config = yaml.safe_load(args.shared_config.read_text(encoding="utf-8"))
    core, _, _ = metadata_gate(
        pipeline, project, args.aux1_state, args.preceding_state, shared_config
    )
    _, selections = verify_selection_set(out)
    for stage in ("signal_evaluation", "downstream"):
        verify_artifact_manifest(out / stage)
    fit_contract = read_json(out / "fit_select/contract.json")
    contract = {
        "schema_version": 1,
        "analysis_profile": PROFILE,
        "pipeline_sha256": sha(pipeline / "pipeline.json"),
        "selection_set_sha256": sha(out / "fit_select/selection_set.json"),
        "config_sha256": sha(args.config),
        "fit_contract_sha256": digest(fit_contract),
        "script_sha256": sha(Path(__file__)),
        "normalized_state_script_sha256": sha(
            Path(__file__).with_name("reviewer_shared_classical.py")
        ),
        "aux1_state_sha256": sha(args.aux1_state),
        "aux2_state_sha256": sha(args.preceding_state),
        "upstream_manifests": {
            stage: sha(out / stage / "artifact_manifest.json")
            for stage in ("fit_select", "signal_evaluation", "downstream")
        },
        "config": config,
        "scope": TIMING_SCOPE,
    }
    destination = out / "cpu_latency"
    if destination.exists():
        if not args.resume:
            raise FileExistsError("Completed CPU output requires explicit --resume")
        verify_artifact_manifest(destination, contract_sha=digest(contract))
        return read_json(destination / "stage_summary.json")
    if any(out.glob("cpu_latency.partial.*")):
        raise RuntimeError("Existing CPU partial evidence requires inspection; no silent restart")
    staging = out / ("cpu_latency.partial." + uuid.uuid4().hex)
    staging.mkdir()
    write_once(staging / "contract.json", contract)
    banks, references = _load_bank_states(core, pipeline, project)
    write_once(staging / "bank_source_references.json", references)
    records, shared, normalized = [], {}, {}
    for protocol in PROTOCOLS:
        shared[protocol] = selections[protocol]["states"]
        normalized[protocol] = {}
        for method, state in shared[protocol].items():
            path = out / "fit_select" / protocol / f"{method}.state.json"
            if read_json(path) != state:
                raise ValueError("Public state file differs from frozen selection state")
            records.append(
                {
                    "protocol": protocol,
                    "canonical_method": method,
                    "information_scope": SCOPE,
                    "variant": "shared",
                    "state": state,
                    "state_path": path.relative_to(out).as_posix(),
                    "state_file_sha256": sha(path),
                    "serialized_state_bytes": path.stat().st_size,
                }
            )
        for pair in banks[protocol]["saved"]["pairs"]:
            pair_key = digest(pair["identity"])
            if pair_key in normalized[protocol]:
                raise ValueError("Duplicate original physical bank pair")
            normalized[protocol][pair_key] = {
                "identity": pair["identity"],
                "states": normalize_bank_pair(protocol, pair),
            }
        # Preserve every family/seed/context reference in the audit, then deduplicate only accounting.
        for reference in (ref for ref in references if ref["protocol"] == protocol):
            for pair in normalized[protocol].values():
                for method, state in pair["states"].items():
                    records.append(
                        {
                            "protocol": protocol,
                            "canonical_method": method,
                            "information_scope": "known_device_bank",
                            "variant": "bank",
                            "pair_identity": pair["identity"],
                            "state": state,
                            "state_path": reference["state_path"],
                            "state_file_sha256": reference["sha256"],
                            "serialized_state_bytes": banks[protocol]["serialized_bytes"],
                        }
                    )
    inventory, totals = parameter_tables(records)
    proofs, schedulers, summaries, raw, workloads, aliases = [], {}, [], [], [], []
    measured = {}

    def measure(states, pools, history, metadata):
        key = digest(
            {
                "states": [item["numeric_sha256"] for item in states],
                "state_contents": [digest(item) for item in states],
                "inputs": [array_sha(pool) for pool in pools],
                "history": history,
            }
        )
        if key not in measured:
            assert_no_competing_experiments()
            result = time_workload(
                [compile_state(item) for item in states],
                pools,
                history=history,
                **{name: config[name] for name in ("warmup", "iterations", "repeats")},
            )
            identifier = "timing_" + key[:24]
            measured[key] = {
                **result,
                "timing_id": identifier,
                "input_pool_sha256": digest([array_sha(pool) for pool in pools]),
                "numeric_state_sha256s": [state["numeric_sha256"] for state in states],
                "state_content_sha256s": [digest(state) for state in states],
                "input_pool_sha256s": [array_sha(pool) for pool in pools],
                "resident_input_shapes": [list(pool.shape) for pool in pools],
                "history_samples": history,
            }
            for index, ms in enumerate(result["raw_wall_ms"]):
                raw.append(
                    {
                        "timing_id": identifier,
                        "measurement": "call_wall_ms",
                        "repeat": index // config["iterations"],
                        "iteration": index % config["iterations"],
                        "value": ms,
                        "unit": "ms",
                    }
                )
            for repeat, seconds in enumerate(result["raw_throughput_loop_seconds"]):
                raw.append(
                    {
                        "timing_id": identifier,
                        "measurement": "independent_throughput_loop",
                        "repeat": repeat,
                        "iteration": None,
                        "value": seconds,
                        "unit": "s",
                    }
                )
        result = measured[key]
        return {
            **metadata,
            **{
                key: value
                for key, value in result.items()
                if key not in ("raw_wall_ms", "raw_throughput_loop_seconds")
            },
            "scope": TIMING_SCOPE,
            "dtype": "float64",
            "device": "cpu",
            "history_samples": history,
            "numeric_state_sha256s_json": json.dumps([state["numeric_sha256"] for state in states]),
            "state_content_sha256s_json": json.dumps([digest(state) for state in states]),
            "resident_input_shapes_json": json.dumps([list(pool.shape) for pool in pools]),
            "timing_alias": any(
                row.get("timing_id") == result["timing_id"] for row in summaries + workloads
            ),
        }

    with single_thread_runtime() as runtime:
        runtime["workload_quiet_check"] = assert_no_competing_experiments()
        write_once(staging / "runtime.json", runtime)
        for protocol in PROTOCOLS:
            frames, _ = load_roles(fit_contract["manifests"][protocol], ("train",), project)
            train = frames["train"]
            if frame_sha(train) != selections[protocol]["train_row_sha256"]:
                raise ValueError("Latency TRAIN physical rows differ from actual fit")
            pair_frames = {
                digest(dict(zip(IDENTITY, map(str, identity), strict=True))): group.copy()
                for identity, group in train.groupby(IDENTITY, sort=True)
            }
            if set(pair_frames) != set(normalized[protocol]):
                raise ValueError("TRAIN scheduler and original bank pair sets differ")
            all_states = [
                *shared[protocol].values(),
                *[
                    state
                    for pair in normalized[protocol].values()
                    for state in pair["states"].values()
                ],
            ]
            history = max(
                max(
                    (state["feature_spec"]["taps"] or 1) - 1,
                    int(state["feature_spec"]["include_delta"]),
                )
                for state in all_states
            )
            for method in METHODS:
                for pair_key, group in pair_frames.items():
                    for scope, state in (
                        (SCOPE, shared[protocol][method]),
                        ("known_device_bank", normalized[protocol][pair_key]["states"][method]),
                    ):
                        proofs.append(
                            {
                                "protocol": protocol,
                                "canonical_method": method,
                                "information_scope": scope,
                                "pair_key": pair_key,
                                "numeric_sha256": state["numeric_sha256"],
                                "proof": verify_kernel(state, group),
                            }
                        )
            for window in config["window_sizes"]:
                pools = {}
                for pair_key, group in pair_frames.items():
                    pool, rows = make_window_pool(
                        group, window, history, max(config["batch_sizes"]), config["seed"]
                    )
                    pools[pair_key] = pool
                    schedulers[f"{protocol}/{window}/{pair_key}"] = {
                        "pair_identity": normalized[protocol][pair_key]["identity"],
                        "physical_train_sha256": frame_sha(group),
                        "pool_sha256": array_sha(pool),
                        "batch_input_sha256": {
                            str(batch): array_sha(pool[:batch]) for batch in config["batch_sizes"]
                        },
                        "rows": rows,
                    }
                keys = sorted(pools)
                for batch in config["batch_sizes"]:
                    resident = [pools[key][:batch] for key in keys]
                    for method in METHODS:
                        common = {
                            "protocol": protocol,
                            "canonical_method": method,
                            "batch_size_per_pair": batch,
                            "window_size": window,
                            "n": 1,
                            "sd": None,
                        }
                        summaries.append(
                            measure(
                                [shared[protocol][method]],
                                [resident[0]],
                                history,
                                {
                                    **common,
                                    "information_scope": SCOPE,
                                    "variant": "shared",
                                    "pair_identity_json": json.dumps(
                                        normalized[protocol][keys[0]]["identity"], sort_keys=True
                                    ),
                                    "measurement_kind": "per_state",
                                },
                            )
                        )
                        for key, pool in zip(keys, resident, strict=True):
                            summaries.append(
                                measure(
                                    [normalized[protocol][key]["states"][method]],
                                    [pool],
                                    history,
                                    {
                                        **common,
                                        "information_scope": "known_device_bank",
                                        "variant": "bank",
                                        "pair_identity_json": json.dumps(
                                            normalized[protocol][key]["identity"], sort_keys=True
                                        ),
                                        "measurement_kind": "per_state",
                                    },
                                )
                            )
                        for scope, states, variant in (
                            (SCOPE, [shared[protocol][method]] * len(keys), "shared"),
                            (
                                "known_device_bank",
                                [normalized[protocol][key]["states"][method] for key in keys],
                                "bank",
                            ),
                        ):
                            row = measure(
                                states,
                                resident,
                                history,
                                {
                                    **common,
                                    "information_scope": scope,
                                    "variant": variant,
                                    "measurement_kind": "full_pair_workload",
                                    "scheduler_sha256": digest(
                                        [schedulers[f"{protocol}/{window}/{key}"] for key in keys]
                                    ),
                                },
                            )
                            workloads.append(row)
                            if row["timing_alias"]:
                                aliases.append(
                                    {
                                        "protocol": protocol,
                                        "canonical_method": method,
                                        "information_scope": scope,
                                        "timing_id": row["timing_id"],
                                        "reason": "exact_numeric_state_and_identical_resident_input_scheduler",
                                    }
                                )
    write_once(staging / "kernel_proofs.json", proofs)
    write_once(staging / "train_window_schedulers.json", schedulers)
    write_once(staging / "timing_aliases.json", aliases)
    write_once(staging / "raw_timing_records.json", measured)
    return finish_stage(
        staging,
        destination,
        {
            "stage": "cpu_latency",
            "contract_sha256": digest(contract),
            "selection_set_sha256": contract["selection_set_sha256"],
            "scope": TIMING_SCOPE,
            "counts": {
                "shared_protocols": len(shared),
                "shared_logical_controls": sum(map(len, shared.values())),
                "paired_final_mappings": len(references),
                "unique_timing_records": len(measured),
            },
            "parameter_interpretation": "stored_learned_scalars_including_intercept_not_statistical_DoF_or_torch_trainable_parameters",
        },
        {
            "state_inventory": pd.DataFrame(inventory),
            "bank_parameter_totals": pd.DataFrame(totals),
            "latency_summary": pd.DataFrame(summaries),
            "latency_raw": pd.DataFrame(raw),
            "workload_latency": pd.DataFrame(workloads),
        },
    )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "pipeline-root", "aux1-state", "preceding-state", "out", "config"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument(
        "--shared-config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/reviewer_shared_classical.yaml",
    )
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"status": result["status"], "stage": result["stage"]}), flush=True)
