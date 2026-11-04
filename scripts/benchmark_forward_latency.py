"""Measure actual checkpoint FP32 forward latency on seeded synthetic windows.

Main latency excludes checkpoint loading, normalization and host/device copies.
CUDA records synchronized per-call host and CUDA-event times, plus a separate
whole-loop throughput measurement without per-call synchronization. Optional
end-to-end timing includes an unpinned CPU input copy and CPU output copy, but
still excludes normalization. Results describe this runtime, not deployment
guarantees or confidence intervals over independent recordings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.models import make_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def latency_statistics(values: list[float]) -> dict:
    """Summarize milliseconds with linear percentiles and sample SD (ddof=1)."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Latency values must be a nonempty finite nonnegative vector")
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "sample_sd_ms": float(array.std(ddof=1)) if array.size > 1 else None,
        "p50_ms": float(np.percentile(array, 50, method="linear")),
        "p95_ms": float(np.percentile(array, 95, method="linear")),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def _normalized_remote(value: str) -> PurePosixPath:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if ".." in path.parts:
        raise ValueError("Source identities must not contain parent traversal")
    return path


def discover_pipeline(root: Path, previous_root: Path | None = None) -> list[dict]:
    """Map exact pipeline identities, never matching a run by basename alone."""
    root = root.resolve()
    pipeline = json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
    if pipeline.get("status") != "complete" or pipeline.get("smoke", False):
        raise ValueError("A completed nonsmoke pipeline is required")
    result = []
    for entry in pipeline["runs"]:
        if entry.get("status") != "evaluated":
            raise ValueError("Pipeline contains an unevaluated run")
        remote_run = _normalized_remote(entry["run_dir"])
        direct = Path(str(entry["run_dir"]))
        if direct.is_dir():
            run = direct.resolve()
            mapping = "exact_existing_run_path"
        else:
            if entry.get("reused"):
                if previous_root is None:
                    raise ValueError("Missing --previous-root for unavailable reused run paths")
                remote_prefix = _normalized_remote(entry["source_pipeline"]).parent
                local_prefix = previous_root.resolve()
            else:
                config = _normalized_remote(entry["config"])
                if config.parent.name != "configs":
                    raise ValueError(
                        "Cannot establish exact current pipeline prefix from frozen config"
                    )
                remote_prefix = config.parent.parent
                local_prefix = root
            try:
                relative = remote_run.relative_to(remote_prefix)
            except ValueError as exc:
                raise ValueError(
                    f"Run identity differs from exact pipeline prefix: {remote_run}"
                ) from exc
            if not relative.parts or relative.parts[0] != "experiments":
                raise ValueError(
                    "Mapped run must be inside the declared pipeline experiments prefix"
                )
            run = local_prefix.joinpath(*relative.parts)
            mapping = f"{remote_prefix} -> {local_prefix}"
        checkpoint = run / "best.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        source_checkpoint = str(remote_run / "best.ckpt")
        expected = entry.get("artifacts", {}).get("sha256", {}).get(source_checkpoint)
        result.append(
            {
                "task": entry["task"],
                "candidate_id": entry["candidate_id"],
                "seed": int(entry["seed"]),
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_expected_sha256": expected,
                "source_run_identity": str(remote_run),
                "path_mapping": mapping,
            }
        )
    _validate_records(result)
    return result


def _validate_records(records: list[dict]) -> None:
    identities = []
    for record in records:
        for key in ("task", "candidate_id", "seed", "checkpoint"):
            if key not in record:
                raise ValueError(f"Checkpoint record lacks {key}")
        if not isinstance(record["candidate_id"], str) or not record["candidate_id"]:
            raise ValueError("candidate_id must be a nonempty string")
        if record["task"] not in {"paired", "navigation"}:
            raise ValueError("task must be paired or navigation")
        if type(record["seed"]) is not int:
            raise ValueError("seed must be an integer")
        if not Path(record["checkpoint"]).is_file():
            raise FileNotFoundError(record["checkpoint"])
        identities.append((record["task"], record["candidate_id"], record["seed"]))
    if not records or len(identities) != len(set(identities)):
        raise ValueError(
            "Checkpoint records must have nonempty distinct task/candidate/seed identities"
        )


def _read_checkpoint_paths(path: Path) -> list[dict]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("--checkpoint-paths must contain a JSON list")
    records = []
    for value in values:
        if isinstance(value, list) and len(value) == 4:
            value = dict(zip(("task", "candidate_id", "seed", "checkpoint"), value, strict=True))
        if not isinstance(value, dict):
            raise ValueError(
                "Each checkpoint record must be an object or [task,candidate_id,seed,path]"
            )
        record = dict(value)
        checkpoint = Path(record["checkpoint"])
        record["checkpoint"] = str(
            (path.parent / checkpoint).resolve()
            if not checkpoint.is_absolute()
            else checkpoint.resolve()
        )
        records.append(record)
    _validate_records(records)
    return records


@contextmanager
def _runtime(device: torch.device, cpu_threads: int):
    """Use the frozen benchmark's strict FP32 policy and restore library state."""
    saved = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "threads": torch.get_num_threads(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "flash": torch.backends.cuda.flash_sdp_enabled(),
        "efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math": torch.backends.cuda.math_sdp_enabled(),
        "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled()
        if hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
        else None,
        "workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    try:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(cpu_threads)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            torch.backends.cuda.enable_cudnn_sdp(False)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA is unavailable; no CPU fallback")
            torch.cuda.set_device(device)
        yield
    finally:
        torch.use_deterministic_algorithms(saved["deterministic"])
        torch.set_num_threads(saved["threads"])
        torch.backends.cudnn.benchmark = saved["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = saved["cudnn_deterministic"]
        torch.backends.cuda.matmul.allow_tf32 = saved["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = saved["cudnn_tf32"]
        torch.backends.cuda.enable_flash_sdp(saved["flash"])
        torch.backends.cuda.enable_mem_efficient_sdp(saved["efficient"])
        torch.backends.cuda.enable_math_sdp(saved["math"])
        if saved["cudnn_sdp"] is not None:
            torch.backends.cuda.enable_cudnn_sdp(saved["cudnn_sdp"])
        if saved["workspace"] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = saved["workspace"]


def _environment(device: torch.device) -> dict:
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    gpu_query = None
    if device.type == "cuda":
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,driver_version,memory.total,utilization.gpu",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            gpu_query = completed.stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            gpu_query = f"unavailable: {type(exc).__name__}"
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_processor": cpu,
        "logical_cpus": os.cpu_count(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "nvidia_smi": gpu_query,
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "precision": "float32, no AMP, no TF32",
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "attention_backend": "math SDPA; flash, memory-efficient and cuDNN SDPA disabled",
        "cudnn_benchmark": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_forward(model, x, device, warmup, iterations, repeats) -> dict:
    wall, event, loop_seconds, loop_event_ms = [], [], [], []
    after_warmup = None
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        _synchronize(device)
        if device.type == "cuda":
            after_warmup = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        for _ in range(repeats):
            current_wall, current_event = [], []
            for _ in range(iterations):
                _synchronize(device)
                before = time.perf_counter()
                if device.type == "cuda":
                    start.record()
                model(x)
                if device.type == "cuda":
                    end.record()
                _synchronize(device)
                current_wall.append((time.perf_counter() - before) * 1000)
                if device.type == "cuda":
                    current_event.append(start.elapsed_time(end))
            wall.append(current_wall)
            if current_event:
                event.append(current_event)
            _synchronize(device)
            before = time.perf_counter()
            if device.type == "cuda":
                start.record()
            for _ in range(iterations):
                model(x)
            if device.type == "cuda":
                end.record()
            _synchronize(device)
            loop_seconds.append(time.perf_counter() - before)
            if device.type == "cuda":
                loop_event_ms.append(start.elapsed_time(end))
    flat_wall = [value for repeat in wall for value in repeat]
    flat_event = [value for repeat in event for value in repeat]
    return {
        "wall": latency_statistics(flat_wall),
        "cuda_event": latency_statistics(flat_event) if flat_event else None,
        "raw_wall_ms": wall,
        "raw_cuda_event_ms": event if event else None,
        "raw_throughput_loop_seconds": loop_seconds,
        "raw_throughput_loop_cuda_event_ms": loop_event_ms if loop_event_ms else None,
        "loop_throughput_windows_per_second": x.shape[0] * iterations * repeats / sum(loop_seconds),
        "loop_cuda_event_windows_per_second": x.shape[0]
        * iterations
        * repeats
        * 1000
        / sum(loop_event_ms)
        if loop_event_ms
        else None,
        "isolated_wall_windows_per_second": x.shape[0] * 1000 / np.mean(flat_wall),
        "gpu_after_warmup_allocated_bytes": after_warmup,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
    }


def _time_end_to_end(model, cpu_input, device, warmup, iterations, repeats) -> dict:
    times = []
    with torch.inference_mode():
        for index in range(warmup + iterations * repeats):
            _synchronize(device)
            before = time.perf_counter()
            output = model(cpu_input.to(device)).cpu()
            _synchronize(device)
            elapsed = (time.perf_counter() - before) * 1000
            del output
            if index >= warmup:
                times.append(elapsed)
    return {
        "wall": latency_statistics(times),
        "raw_wall_ms": times,
        "scope": "unpinned normalized CPU input -> device forward -> CPU output; normalization excluded",
    }


def _benchmark_record(record: dict, args, device: torch.device) -> list[dict]:
    path = Path(record["checkpoint"])
    digest = _sha256(path)
    if record.get("checkpoint_expected_sha256") and digest != record["checkpoint_expected_sha256"]:
        raise ValueError(f"Checkpoint SHA256 differs from pipeline artifact record: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_config = checkpoint.get("model_config", config["model"])
    if model_config != config["model"] or int(config["seed"]) != record["seed"]:
        raise ValueError(f"Checkpoint model/seed identity differs from declared record: {path}")
    model = make_model(model_config).float().eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    checkpoint_metadata = {
        key: checkpoint.get(key)
        for key in (
            "git_commit",
            "best_epoch",
            "best_metric",
            "checkpoint_metric",
            "target_mode",
            "target_layout",
            "prediction_head",
            "label_mode",
            "feature_columns",
            "target_columns",
        )
    }
    del checkpoint
    in_channels = int(model_config["in_channels"])
    out_channels = int(model_config["out_channels"])
    length = int(config["data"]["window_size"])
    if device.type == "cuda":
        torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    model = model.to(device)
    _synchronize(device)
    resident = torch.cuda.memory_allocated(device) - baseline if device.type == "cuda" else None
    source_paths = {
        Path(inspect.getfile(type(module))).resolve()
        for module in model.modules()
        if type(module).__module__.startswith("virtual_imu")
    }
    common = {
        **record,
        "checkpoint_sha256": digest,
        "checkpoint_metadata": checkpoint_metadata,
        "config_snapshot": config,
        "config_snapshot_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "model_config": model_config,
        "model_factory_sha256": _sha256(Path(inspect.getfile(make_model))),
        "model_source_sha256": {str(source): _sha256(source) for source in sorted(source_paths)},
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "buffers": sum(b.numel() for b in model.buffers()),
        "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
        "input_channels": in_channels,
        "output_channels": out_channels,
        "window_size": length,
        "device": str(device),
        "precision": "float32",
        "gpu_baseline_allocated_bytes": baseline,
        "gpu_model_resident_increment_bytes": resident,
        "forward_scope": "model.eval with inference_mode; normalized synthetic input resident on device; all outputs; copies excluded",
    }
    results = []
    for batch in args.batch_sizes:
        generator = torch.Generator(device="cpu").manual_seed(args.input_seed)
        cpu_input = torch.randn(
            batch, in_channels, length, generator=generator, dtype=torch.float32
        )
        x = cpu_input.to(device)
        with torch.inference_mode():
            output = model(x)
            if not isinstance(output, torch.Tensor) or output.shape != (
                batch,
                out_channels,
                length,
            ):
                raise ValueError(f"Unexpected model output shape: {path}")
            if output.dtype != torch.float32 or not torch.isfinite(output).all().item():
                raise ValueError(f"Nonfinite/non-FP32 model output: {path}")
            output_shape = list(output.shape)
            del output
        before_forward = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
        measured = _time_forward(model, x, device, args.warmup, args.iterations, args.repeats)
        measured["gpu_forward_peak_increment_bytes"] = (
            measured["gpu_peak_allocated_bytes"] - measured["gpu_after_warmup_allocated_bytes"]
            if device.type == "cuda"
            else None
        )
        results.append(
            {
                **common,
                "batch_size": batch,
                "input_shape": list(x.shape),
                "output_shape": output_shape,
                "input_bytes": x.numel() * x.element_size(),
                "input_seed": args.input_seed,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "gpu_before_forward_allocated_bytes": before_forward,
                **measured,
                "end_to_end": _time_end_to_end(
                    model, cpu_input, device, args.warmup, args.iterations, args.repeats
                )
                if args.end_to_end
                else None,
            }
        )
        del x
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _flat_record(record: dict) -> dict:
    flat = {
        key: record[key]
        for key in (
            "task",
            "candidate_id",
            "seed",
            "device",
            "precision",
            "batch_size",
            "parameters",
            "trainable_parameters",
            "parameter_bytes",
            "buffers",
            "buffer_bytes",
            "input_channels",
            "output_channels",
            "window_size",
            "loop_throughput_windows_per_second",
            "gpu_peak_allocated_bytes",
            "gpu_model_resident_increment_bytes",
            "gpu_forward_peak_increment_bytes",
            "checkpoint",
            "checkpoint_sha256",
        )
    }
    for name in ("wall", "cuda_event"):
        for key in ("mean_ms", "sample_sd_ms", "p50_ms", "p95_ms", "min_ms"):
            flat[f"{name}_{key}"] = record[name][key] if record[name] else None
    return flat


def _seed_summary(records: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for record in records:
        groups[
            (record["task"], record["candidate_id"], record["device"], record["batch_size"])
        ].append(_flat_record(record))
    results = []
    for (task, candidate, device, batch), rows in sorted(groups.items()):
        result = {
            "task": task,
            "candidate_id": candidate,
            "device": device,
            "batch_size": batch,
            "seed_count": len(rows),
            "statistic_scope": "mean and sample SD across optimization-seed checkpoints; not recording CI",
        }
        for metric in (
            "parameters",
            "wall_mean_ms",
            "wall_p50_ms",
            "wall_p95_ms",
            "cuda_event_mean_ms",
            "cuda_event_p50_ms",
            "cuda_event_p95_ms",
            "loop_throughput_windows_per_second",
        ):
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            result[f"{metric}_mean"] = float(np.mean(values)) if values else None
            result[f"{metric}_sample_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        results.append(result)
    return results


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _raw_timing_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        identity = {
            key: record[key] for key in ("task", "candidate_id", "seed", "device", "batch_size")
        }
        for repeat, times in enumerate(record["raw_wall_ms"]):
            for iteration, wall in enumerate(times):
                rows.append(
                    {
                        **identity,
                        "repeat": repeat,
                        "iteration": iteration,
                        "wall_ms": wall,
                        "cuda_event_ms": record["raw_cuda_event_ms"][repeat][iteration]
                        if record["raw_cuda_event_ms"]
                        else None,
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pipeline-root", type=Path)
    source.add_argument("--checkpoint-paths", type=Path)
    parser.add_argument("--previous-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+")
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--input-seed", type=int, default=20261024)
    parser.add_argument("--end-to-end", action="store_true")
    args = parser.parse_args(argv)
    args.batch_sizes = args.batch_sizes or ([1] if args.device == "cpu" else [1, 128])
    if args.cpu_threads is None:
        args.cpu_threads = 1 if args.device == "cpu" else 4
    if (
        args.warmup < 0
        or min(args.iterations, args.repeats, args.cpu_threads, *args.batch_sizes) < 1
    ):
        parser.error(
            "warmup must be nonnegative; iterations/repeats/threads/batch sizes must be positive"
        )
    if len(args.batch_sizes) != len(set(args.batch_sizes)):
        parser.error("batch sizes must be distinct")
    print(
        f"READS: explicit checkpoints from {args.pipeline_root or args.checkpoint_paths}; seeded synthetic normalized inputs",
        flush=True,
    )
    print(f"WRITES: new immutable latency artifacts under {args.out}", flush=True)
    print(
        "SKIPS: datasets, normalization, training, accuracy evaluation, checkpoint changes, outputs/latest, loading/transfers in main timing",
        flush=True,
    )
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite latency output: {args.out}")
    records = (
        discover_pipeline(args.pipeline_root, args.previous_root)
        if args.pipeline_root
        else _read_checkpoint_paths(args.checkpoint_paths)
    )
    args.out.mkdir(parents=True, exist_ok=False)
    raw_dir = args.out / "records"
    raw_dir.mkdir()
    source_path = (
        args.pipeline_root / "pipeline.json" if args.pipeline_root else args.checkpoint_paths
    )
    source = {"path": str(source_path.resolve()), "sha256": _sha256(source_path)}
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    results = []
    try:
        with _runtime(device, args.cpu_threads):
            environment = _environment(device)
            _write_json(args.out / "environment.json", environment)
            for record in records:
                print(
                    f"READS: {record['checkpoint']} ({record['task']}/{record['candidate_id']}/seed={record['seed']})",
                    flush=True,
                )
                for measured in _benchmark_record(record, args, device):
                    _write_json(raw_dir / f"{len(results):04d}.json", measured)
                    results.append(measured)
                    print(
                        f"MEASURED: batch={measured['batch_size']} wall_p50_ms={measured['wall']['p50_ms']:.6f} windows_per_second={measured['loop_throughput_windows_per_second']:.3f}",
                        flush=True,
                    )
        summary = _seed_summary(results)
        _write_csv(args.out / "latency_per_seed.csv", [_flat_record(record) for record in results])
        _write_csv(args.out / "latency_raw_timings.csv", _raw_timing_rows(results))
        _write_csv(args.out / "latency_seed_summary.csv", summary)
        _write_json(args.out / "latency_seed_summary.json", summary)
        _write_json(
            args.out / "latency.json",
            {
                "status": "complete",
                "source": source,
                "environment": environment,
                "script_sha256": _sha256(Path(__file__)),
                "records": results,
                "seed_summary": summary,
                "timing_scope": "resident-input full model forward, eval/inference_mode, strict FP32; warmup excluded; no transfers/normalization",
                "limitations": [
                    "Synthetic standard-normal inputs; no accuracy or data-loading measurement.",
                    "P95 is empirical across repeated calls; seed SD is not a recording-level confidence interval.",
                    "Host wall samples include synchronization/event overhead on CUDA; throughput uses separate whole-loop timing.",
                    "No exclusive GPU lock is acquired; inspect recorded environment/utilization and run on an idle device.",
                ],
            },
        )
    except Exception as exc:
        _write_json(
            args.out / "failure.json",
            {
                "status": "failed",
                "error": repr(exc),
                "completed_records": len(results),
                "source": source,
            },
        )
        raise
    print(
        f"WRITTEN: {args.out / 'latency.json'}; {len(results)} checkpoint/batch records", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
