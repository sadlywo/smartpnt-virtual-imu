"""Audit every final reviewer checkpoint, then time unique actual architectures.

Timing uses seed 20261010 by a fixed identity rule, never accuracy. It runs only
after the nonsmoke core pipeline has 93 selected cells and 558 final evaluations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import io
import json
import math
import re
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.models import make_model
from virtual_imu.training.reviewer_hpo import _verify_selection

REPRESENTATIVE_SEED = 20261010
FINAL_SEEDS = {20261010, 20261011, 20261012}
HISTORICAL = Path(__file__).with_name("benchmark_forward_latency.py")


def _measurement_api():
    spec = importlib.util.spec_from_file_location("reviewer_forward_latency_api", HISTORICAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _write_once(path: Path, value) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Preserving a different existing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def _state_write(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    text = buffer.getvalue()
    if path.exists():
        if path.read_text(encoding="utf-8").replace("\r\n", "\n") != text.replace("\r\n", "\n"):
            raise FileExistsError(f"Preserving a different existing output table: {path}")
        return
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)


def _finite(value, where: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite(item, f"{where}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite metadata: {where}")


def _posix(value: str) -> PurePosixPath:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if ".." in path.parts:
        raise ValueError("Source artifact identities may not contain parent traversal")
    return path


def _source_mapper(root: Path, state: dict):
    prefixes = set()
    for job in state["jobs"].values():
        config = _posix(job["config"])
        if config.parent.name != "configs":
            raise ValueError("Frozen job configs must establish an exact pipeline prefix")
        prefixes.add(config.parent.parent)
    if len(prefixes) != 1:
        raise ValueError("Pipeline job configs disagree on the exact source prefix")
    source = next(iter(prefixes))

    def mapped(value: str) -> Path:
        try:
            relative = _posix(value).relative_to(source)
        except ValueError as error:
            raise ValueError(f"Artifact is outside the exact pipeline source prefix: {value}") from error
        if not relative.parts or relative.parts[0] not in {"configs", "experiments", "evaluations", "selections", "classical"}:
            raise ValueError("Unexpected pipeline artifact location")
        path = root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(root):
            raise ValueError("Resolved pipeline artifact escapes the supplied root")
        return path

    return str(source), mapped


def _verify_hash(path: Path, expected: str) -> None:
    if _sha(path) != expected:
        raise ValueError(f"Artifact SHA256 changed: {path}")


def _model_source_hashes(model, source_contract: dict) -> dict:
    paths = {Path(inspect.getfile(make_model)).resolve()}
    paths.update(Path(inspect.getfile(type(module))).resolve() for module in model.modules() if type(module).__module__.startswith("virtual_imu"))
    result = {}
    for path in sorted(paths):
        parts = _posix(str(path)).parts
        anchors = [index for index in range(len(parts) - 1) if parts[index:index + 2] == ("src", "virtual_imu")]
        if not anchors:
            raise ValueError("Cannot bind model implementation to its full package path")
        relative = PurePosixPath(*parts[anchors[-1]:])
        expected = [digest for identity, digest in source_contract.items() if _posix(identity).parts[-len(relative.parts):] == relative.parts]
        digest = _sha(path)
        if len(expected) != 1 or expected[0] != digest:
            raise ValueError(f"Runtime model source differs from the frozen HPO source: {relative}")
        result[str(relative)] = digest
    return result


def _checkpoint_record(job: dict, cell: dict, mapped, source_contract: dict) -> dict:
    directory = mapped(job["run_dir"])
    frozen = mapped(job["config"])
    _verify_hash(frozen, job["config_sha256"])
    for name, expected in job["artifact_sha256"].items():
        _verify_hash(directory / name, expected)
    checkpoint_path = directory / "best.ckpt"
    _verify_hash(checkpoint_path, job["checkpoint_sha256"])
    config = yaml.safe_load(frozen.read_text(encoding="utf-8"))
    _finite(config, "checkpoint_config")
    if yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8")) != config:
        raise ValueError("Run config differs from frozen source config")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = config["model"]
    if checkpoint.get("config") != config or checkpoint.get("model_config") != model_config:
        raise ValueError("Checkpoint model/config differs from its frozen source")
    metadata = config["reviewer_protocol"]
    variant = job["phase"].removeprefix("final_")
    expected_trial = cell["trials"][0] if variant == "default" else cell["selected"]["trial"]
    if config["seed"] != job["seed"] or metadata["variant"] != variant or metadata["model_family"] != cell["candidate"]["id"] or metadata["protocol_id"] != cell["protocol"]["id"] or metadata["parameters"] != expected_trial["parameters"] or job["trial_id"] != expected_trial["id"]:
        raise ValueError("Final checkpoint identity or default/selected parameters disagree")
    for dotted, expected in expected_trial["parameters"].items():
        actual = metadata.get("loss_profile") if dotted == "loss_profile" else config
        if dotted != "loss_profile":
            for part in dotted.split("."):
                actual = actual[part]
        if actual != expected:
            raise ValueError(f"Frozen model/training parameters were not applied: {dotted}")
    if "window_size" in model_config and int(model_config["window_size"]) != int(config["data"]["window_size"]):
        raise ValueError("Model and input context lengths disagree")
    score, epoch = checkpoint.get("best_metric"), checkpoint.get("best_epoch")
    metric = config["training"]["checkpoint_metric"]
    if not isinstance(score, (float, int)) or not math.isfinite(score) or score < 0 or type(epoch) is not int or not 1 <= epoch <= config["training"]["epochs"] or checkpoint.get("checkpoint_metric") != metric:
        raise ValueError("Invalid finite validation checkpoint identity")
    with (directory / "metrics.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    values = [float(row[metric]) for row in history]
    epochs = [int(float(row["epoch"])) for row in history]
    if not values or not all(math.isfinite(value) for value in values) or epochs != sorted(set(epochs)) or epoch not in epochs or not math.isclose(min(values), score, rel_tol=1e-6, abs_tol=1e-9) or not math.isclose(values[epochs.index(epoch)], score, rel_tol=1e-6, abs_tol=1e-9) or not math.isclose(job["score"], score, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Validation checkpoint/history/ledger disagree")
    weights = checkpoint.get("model_state_dict")
    if not isinstance(weights, dict) or not weights or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() or (value.is_floating_point() and value.dtype != torch.float32) for value in weights.values()):
        raise ValueError("Checkpoint state must contain finite FP32 floating tensors")
    model = make_model(model_config).float().eval()
    model.load_state_dict(weights, strict=True)
    sources = _model_source_hashes(model, source_contract)
    task, window = cell["protocol"]["task"], int(config["data"]["window_size"])
    if model_config["in_channels"] != 6 or model_config["out_channels"] != (6 if task == "paired" else 3) or (task == "paired" and window not in {80, 160, 320}) or (task == "navigation" and window != 80):
        raise ValueError("Final model/task/input shape violates the reviewer contract")
    schema = [[name, list(value.shape), str(value.dtype)] for name, value in model.state_dict().items()]
    architecture = {"task": task, "model_config": model_config, "input_channels": 6, "output_channels": model_config["out_channels"], "window_size": window, "state_schema": schema, "model_source_sha256": sources}
    record = {
        "entry_id": job["id"], "protocol_id": cell["protocol"]["id"], "task": task,
        "model_family": cell["candidate"]["id"], "variant": variant, "seed": job["seed"],
        "trial_id": job["trial_id"], "architecture_id": _digest(architecture),
        "architecture": architecture, "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": job["checkpoint_sha256"], "source_run_identity": job["run_dir"],
        "source_config_identity": job["config"], "frozen_config": str(frozen),
        "frozen_config_sha256": job["config_sha256"], "config_snapshot_sha256": _digest(config),
        "validation_metric": metric, "best_epoch": epoch, "best_metric": float(score),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "parameter_bytes": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()),
        "buffers": sum(buffer.numel() for buffer in model.buffers()),
        "buffer_bytes": sum(buffer.numel() * buffer.element_size() for buffer in model.buffers()),
        "checkpoint_state_schema_sha256": _digest(schema), "input_channels": 6,
        "output_channels": model_config["out_channels"], "window_size": window,
    }
    del model, checkpoint
    return record


def validate_reviewer_pipeline(root: Path, *, expected_cells: int = 93, expected_evaluations: int = 558) -> dict:
    """Audit complete core artifacts; smaller expected counts are for fixtures only."""
    root = root.resolve()
    pipeline_path = root / "pipeline.json"
    state = json.loads(pipeline_path.read_text(encoding="utf-8"))
    _finite(state, "pipeline")
    cells = state["cells"]
    if state.get("status") != "complete" or state.get("smoke") or len(cells) != expected_cells or any("selected" not in cell for cell in cells.values()):
        raise ValueError("Latency requires the completed nonsmoke core and every selected cell")
    if set(state["final_seeds"]) != FINAL_SEEDS or state["final_variants"] != ["default", "tuned"]:
        raise ValueError("Latency requires all three fixed seeds and both final variants")
    final = [job for job in state["jobs"].values() if job["phase"] in {"final_default", "final_tuned"}]
    if len(final) != expected_evaluations or state["counts"]["final_evaluated"] != expected_evaluations or state["counts"]["expected_final_runs"] != expected_evaluations or state["counts"]["selected_cells"] != expected_cells or any(job["status"] != "complete" or job.get("evaluation_status") != "complete" for job in final):
        raise ValueError("Latency requires complete final checkpoint/evaluation coverage")
    source_prefix, mapped = _source_mapper(root, state)
    frozen_matrix = yaml.safe_load((root / "matrix_config.yaml").read_text(encoding="utf-8"))
    if frozen_matrix != state["contract"]["config"]:
        raise ValueError("Frozen matrix and pipeline contract disagree")
    expected_roster = {f"{protocol['id']}/{candidate['id']}" for protocol in frozen_matrix["protocols"] for candidate in frozen_matrix["candidates"] if protocol["task"] in candidate["tasks"]}
    if set(cells) != expected_roster:
        raise ValueError("Completed cells differ from the full declared model/protocol roster")
    inputs = {str(pipeline_path): _sha(pipeline_path), str(root / "matrix_config.yaml"): _sha(root / "matrix_config.yaml")}
    records = []
    for cell_id in cells:
        expected = {(phase, seed) for phase in ("final_default", "final_tuned") for seed in FINAL_SEEDS}
        observed = [(job["phase"], job["seed"]) for job in final if job["cell"] == cell_id]
        if len(observed) != len(expected) or set(observed) != expected:
            raise ValueError("A model/protocol cell lacks one exact final variant/seed identity")
        _verify_selection(cell_id, state, root)
        selection_path = root / "selections" / f"{cell_id.replace('/', '__')}.json"
        inputs[str(selection_path)] = _sha(selection_path)
    for job in sorted(final, key=lambda item: item["id"]):
        cell = cells[job["cell"]]
        report_path = root / "evaluations" / f"{job['id']}.json"
        _verify_hash(report_path, job["evaluation_sha256"])
        actual_inventory = set()
        run_dir = mapped(job["run_dir"])
        for identity, digest in job["evaluation_artifact_sha256"].items():
            path = mapped(identity)
            _verify_hash(path, digest)
            inputs[str(path)] = digest
            actual_inventory.add(path.resolve())
        decoded_inventory = {path.resolve() for path in run_dir.rglob("*") if path.is_file() and path.suffix.lower() in {".parquet", ".json", ".csv"}}
        if not decoded_inventory.issubset(actual_inventory) or report_path.resolve() not in actual_inventory:
            raise ValueError("Final evaluation artifact inventory is incomplete")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _finite(report, "evaluation")
        if report["job_id"] != job["id"] or report["variant"] != job["phase"].removeprefix("final_") or report["model_family"] != cell["candidate"]["id"] or report["checkpoint_sha256"] != job["checkpoint_sha256"] or report["run_dir"] != job["run_dir"]:
            raise ValueError("Final evaluation summary identity disagrees with the pipeline")
        record = _checkpoint_record(job, cell, mapped, state["contract"]["source_sha256"])
        metrics = report["metrics"]
        if metrics.get("training_run") != job["run_dir"] or metrics.get("task") != record["task"] or metrics.get("seed") != record["seed"] or metrics.get("reviewer_protocol", {}).get("variant") != record["variant"]:
            raise ValueError("Actual evaluation metrics identity differs from the checkpoint")
        record["evaluation_summary"] = str(report_path)
        record["evaluation_sha256"] = job["evaluation_sha256"]
        records.append(record)
    groups = {}
    for record in records:
        group = groups.setdefault(record["architecture_id"], {"id": record["architecture_id"], "architecture": record["architecture"], "entry_ids": [], "representative_entry_id": None})
        group["entry_ids"].append(record["entry_id"])
    lookup = {record["entry_id"]: record for record in records}
    for group in groups.values():
        representatives = [lookup[identity] for identity in group["entry_ids"] if lookup[identity]["seed"] == REPRESENTATIVE_SEED]
        if not representatives:
            raise ValueError("Architecture lacks the fixed representative seed 20261010")
        representative = min(representatives, key=lambda item: (item["protocol_id"], item["variant"], item["entry_id"]))
        group["representative_entry_id"] = representative["entry_id"]
        for identity in group["entry_ids"]:
            lookup[identity]["timing_representative_entry_id"] = representative["entry_id"]
    return {"pipeline_root": str(root), "source_pipeline_prefix": source_prefix, "pipeline_sha256": inputs[str(pipeline_path)], "source_artifact_sha256": inputs, "entries": records, "groups": groups, "selection_rule": "seed 20261010, then lexicographic protocol_id/variant/entry_id; never accuracy"}


def _environment_contract(environment: dict) -> dict:
    result = {key: value for key, value in environment.items() if key not in {"created_utc", "nvidia_smi"}}
    query = environment.get("nvidia_smi")
    result["nvidia_smi_identity"] = "\n".join(line.rsplit(",", 1)[0] for line in query.splitlines()) if query else None
    return result


def _validate_measurements(records: list[dict], representative: dict, settings: dict, api) -> None:
    if [record["batch_size"] for record in records] != settings["batch_sizes"]:
        raise ValueError("Architecture timing batch coverage differs from the protocol")
    for record in records:
        _finite(record, "latency")
        if record["checkpoint_sha256"] != representative["checkpoint_sha256"] or record["seed"] != REPRESENTATIVE_SEED or record["parameters"] != representative["parameters"] or record["window_size"] != representative["window_size"] or record["device"] != settings["device"] or record["model_config"] != representative["architecture"]["model_config"] or record["config_snapshot_sha256"] != representative["config_snapshot_sha256"]:
            raise ValueError("Measured checkpoint/architecture identity differs")
        if any(record[key] != settings[key] for key in ("warmup", "iterations", "repeats", "input_seed")) or record["input_shape"] != [record["batch_size"], 6, representative["window_size"]] or record["output_shape"] != [record["batch_size"], representative["output_channels"], representative["window_size"]]:
            raise ValueError("Measured timing protocol or tensor shape differs")
        raw = record["raw_wall_ms"]
        if len(raw) != settings["repeats"] or any(len(repeat) != settings["iterations"] for repeat in raw):
            raise ValueError("Incomplete latency repeat/iteration coverage")
        derived = api.latency_statistics([value for repeat in raw for value in repeat])
        for key in ("count", "p50_ms", "p95_ms", "mean_ms"):
            if not math.isclose(record["wall"][key], derived[key], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Saved latency statistics differ from raw timings")
        if record["loop_throughput_windows_per_second"] <= 0:
            raise ValueError("Nonpositive measured throughput")
        loops = record["raw_throughput_loop_seconds"]
        if len(loops) != settings["repeats"] or any(value <= 0 for value in loops) or not math.isclose(record["loop_throughput_windows_per_second"], record["batch_size"] * settings["iterations"] * settings["repeats"] / sum(loops), rel_tol=1e-10):
            raise ValueError("Throughput disagrees with the separate measured loops")
        if settings["device"].startswith("cuda"):
            event_raw = record["raw_cuda_event_ms"]
            if event_raw is None or len(event_raw) != settings["repeats"] or any(len(repeat) != settings["iterations"] for repeat in event_raw):
                raise ValueError("CUDA event timing coverage is incomplete")
            derived_event = api.latency_statistics([value for repeat in event_raw for value in repeat])
            if record["cuda_event"] != derived_event or record["loop_cuda_event_windows_per_second"] <= 0:
                raise ValueError("CUDA event statistics disagree with raw measurements")


def run_latency(
    pipeline_root: Path, out: Path, *, device: str, resume: bool = False,
    warmup: int = 30, iterations: int = 100, repeats: int = 3,
    cpu_threads: int | None = None, input_seed: int = 20261024,
    expected_cells: int = 93, expected_evaluations: int = 558,
) -> dict:
    """Run independently after core completion; small count overrides serve tests."""
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"} or (resolved.type == "cuda" and resolved.index is None):
        raise ValueError("Use cpu or an explicit CUDA index such as cuda:0")
    if warmup < 0 or min(iterations, repeats, cpu_threads or 1) < 1:
        raise ValueError("Invalid timing budget or CPU thread count")
    out = out.resolve()
    pipeline_root = pipeline_root.resolve()
    if out == pipeline_root or out.is_relative_to(pipeline_root / "experiments"):
        raise ValueError("Timing output must not modify core checkpoint directories")
    if out.exists() and not resume:
        raise FileExistsError(f"Preserving existing latency output: {out}")
    if resume and not (out / "latency_state.json").is_file():
        raise FileNotFoundError("Resume requires an existing latency_state.json")
    print(f"READS: complete core pipeline and every final artifact {pipeline_root}", flush=True)
    print(f"WRITES: independent deduplicated latency artifacts {out}", flush=True)
    print("SKIPS: training-time timing, data loading, accuracy selection, checkpoint changes and normalization/copy costs", flush=True)
    audit = validate_reviewer_pipeline(pipeline_root, expected_cells=expected_cells, expected_evaluations=expected_evaluations)
    api = _measurement_api()
    settings = {"device": str(resolved), "batch_sizes": [1, 128] if resolved.type == "cuda" else [1], "cpu_threads": cpu_threads if cpu_threads is not None else (4 if resolved.type == "cuda" else 1), "warmup": warmup, "iterations": iterations, "repeats": repeats, "input_seed": input_seed, "end_to_end": False}
    contract = {"pipeline_sha256": audit["pipeline_sha256"], "audit_sha256": _digest(audit), "settings": settings, "wrapper_sha256": _sha(Path(__file__)), "measurement_api_sha256": _sha(HISTORICAL), "representative_seed": REPRESENTATIVE_SEED}
    if resume:
        state = json.loads((out / "latency_state.json").read_text(encoding="utf-8"))
        if state["contract"] != contract:
            raise ValueError("Resume source, architecture inventory, implementation or timing protocol changed")
        if any(group["status"] in {"running", "failed"} for group in state["groups"].values()):
            raise RuntimeError("Partial timing is preserved and will not be overwritten; use a new output for replacement")
    else:
        out.mkdir(parents=True, exist_ok=False)
        state = {"schema_version": 1, "created_utc": datetime.now(UTC).isoformat(), "status": "planned", "contract": contract, "groups": {identifier: {"status": "planned"} for identifier in audit["groups"]}}
        _write_once(out / "checkpoint_inventory.json", audit)
        _state_write(out / "latency_state.json", state)
    entries = {entry["entry_id"]: entry for entry in audit["entries"]}
    results = []
    try:
        with api._runtime(resolved, settings["cpu_threads"]):
            environment = api._environment(resolved)
            runtime = _environment_contract(environment)
            if "runtime" in state and state["runtime"] != runtime:
                raise ValueError("Effective latency runtime changed on resume")
            if "runtime" not in state:
                state["runtime"] = runtime
                _write_once(out / "environment.json", environment)
            state["status"] = "measuring"
            _state_write(out / "latency_state.json", state)
            args = SimpleNamespace(**settings)
            for identifier, group in sorted(audit["groups"].items()):
                _verify_hash(pipeline_root / "pipeline.json", audit["pipeline_sha256"])
                status = state["groups"][identifier]
                representative = entries[group["representative_entry_id"]]
                path = out / "architectures" / identifier / "measurements.json"
                if status["status"] == "complete":
                    _verify_hash(path, status["sha256"])
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    _validate_measurements(payload["measurements"], representative, settings, api)
                    results.extend(payload["measurements"])
                    print(f"SKIPS: verified architecture timing {identifier}", flush=True)
                    continue
                if path.parent.exists():
                    raise FileExistsError(f"Preserving partial architecture timing: {path.parent}")
                path.parent.mkdir(parents=True)
                status.update(status="running", started_utc=datetime.now(UTC).isoformat())
                _state_write(out / "latency_state.json", state)
                try:
                    record = {"task": representative["task"], "candidate_id": representative["model_family"], "seed": REPRESENTATIVE_SEED, "checkpoint": representative["checkpoint"], "checkpoint_expected_sha256": representative["checkpoint_sha256"], "architecture_id": identifier, "representative_entry_id": representative["entry_id"]}
                    print(f"READS: fixed-seed architecture representative {record['checkpoint']}", flush=True)
                    measured = api._benchmark_record(record, args, resolved)
                    _verify_hash(pipeline_root / "pipeline.json", audit["pipeline_sha256"])
                    _validate_measurements(measured, representative, settings, api)
                    _write_once(path, {"architecture_id": identifier, "representative_entry_id": representative["entry_id"], "measurements": measured})
                    status.update(status="complete", sha256=_sha(path), completed_utc=datetime.now(UTC).isoformat())
                    results.extend(measured)
                except Exception as error:
                    status.update(status="failed", error=repr(error), traceback=traceback.format_exc())
                    raise
                finally:
                    _state_write(out / "latency_state.json", state)
        for name, digest in state.get("final_artifact_sha256", {}).items():
            _verify_hash(out / name, digest)
        inventory_rows = [{key: value for key, value in entry.items() if key != "architecture"} for entry in audit["entries"]]
        mapping = [{"entry_id": entry["entry_id"], "protocol_id": entry["protocol_id"], "model_family": entry["model_family"], "variant": entry["variant"], "seed": entry["seed"], "architecture_id": entry["architecture_id"], "timing_representative_entry_id": entry["timing_representative_entry_id"]} for entry in audit["entries"]]
        flat = [{"architecture_id": result["architecture_id"], "representative_entry_id": result["representative_entry_id"], **api._flat_record(result), **{key: result[key] for key in ("loop_cuda_event_windows_per_second", "isolated_wall_windows_per_second", "gpu_baseline_allocated_bytes", "gpu_before_forward_allocated_bytes", "gpu_after_warmup_allocated_bytes", "warmup", "iterations", "repeats", "input_seed")}} for result in results]
        raw = [{"architecture_id": result["architecture_id"], "representative_entry_id": result["representative_entry_id"], **row} for result in results for row in api._raw_timing_rows([result])]
        report = {"schema_version": 1, "status": "complete", "source_pipeline": str(pipeline_root), "source_pipeline_sha256": audit["pipeline_sha256"], "entries": len(audit["entries"]), "unique_architectures": len(audit["groups"]), "measured_checkpoint_batch_records": len(results), "selection_rule": audit["selection_rule"], "timing_protocol": settings, "matches_historical_default_budget": warmup == 30 and iterations == 100 and repeats == 3, "records": results, "limitations": ["Seed 20261010 represents each identical full architecture/input shape; other seed checkpoints are inventoried, not timed independently.", "No accuracy or latency result selects the representative checkpoint.", "Empirical p50/p95 and sample SD describe repeated timing calls; they are not optimization-seed or recording confidence intervals.", "Resident normalized synthetic FP32 input, eval/inference_mode; checkpoint loading, preprocessing and copies excluded.", "Throughput uses a separate whole loop; CUDA wall timings include synchronization overhead and CUDA events are reported separately.", "This runtime's latency is not a real-time device sampling-rate or deployment guarantee; no exclusive device lock is acquired."]}
        for name, rows in (("checkpoint_inventory.csv", inventory_rows), ("architecture_mapping.csv", mapping), ("latency_architectures.csv", flat), ("latency_raw_timings.csv", raw)):
            _write_csv(out / name, rows)
        _write_once(out / "architecture_mapping.json", mapping)
        _write_once(out / "latency.json", report)
        state.update(status="complete", final_artifact_sha256={str(path.relative_to(out)): _sha(path) for path in out.rglob("*") if path.is_file() and path.name not in {"latency_state.json", "latency_state.tmp"}})
        _state_write(out / "latency_state.json", state)
    except Exception as error:
        state.update(status="failed", error=repr(error), traceback=traceback.format_exc())
        _state_write(out / "latency_state.json", state)
        raise
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"cpu|cuda:\d+", args.device):
        parser.error("--device must be cpu or an explicit index such as cuda:0")
    report = run_latency(args.pipeline_root, args.out, device=args.device, resume=args.resume, warmup=args.warmup, iterations=args.iterations, repeats=args.repeats, cpu_threads=args.cpu_threads)
    print(f"WRITES: {args.out / 'latency.json'} entries={report['entries']} architectures={report['unique_architectures']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
