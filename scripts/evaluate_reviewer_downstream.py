"""Final-only, immutable independent-GT downstream replay of the reviewer matrix.

No test prediction payload is read until all 93 validation selections and all
558 final evaluations pass their recorded provenance/checksum gate. Four Penn
outer folds are primary; pooled/bs is retained as a separate same-recording view.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from virtual_imu.eval.downstream_attitude import (
    evaluate_attitude_stream,
    evaluate_strapdown_stream,
    source_clock_origin_s,
)
from virtual_imu.eval.formal_evaluation import HIGH, IDENTITY, LOW, VIRTUAL, json_safe
from virtual_imu.eval.reviewer_evaluation import _predict_paired_classical
from virtual_imu.training.reviewer_hpo import _verify_selection


def project_root() -> Path:
    """Auxiliary scripts may live outside the frozen checkout; cwd owns its src."""
    for candidate in (Path.cwd().resolve(), Path(__file__).resolve().parents[1]):
        if (candidate / "src/virtual_imu").is_dir():
            return candidate
    raise ValueError("Run from the frozen training checkout containing src/virtual_imu")


PROJECT = project_root()
UNITS = {
    "penn_loso_af": "af",
    "penn_loso_as": "as",
    "penn_loso_bf": "bf",
    "penn_only": "bs",
    "pooled": "bs",
}
PENN_IDENTITY = {
    "dataset_id": "penncosyvio",
    "low_sensor": "Tango Top IMU",
    "high_sensor": "ADIS16488",
}
AXES = ["gx", "gy", "gz", "ax", "ay", "az"]
KEYS = ["sequence_id", "timestamp_s"]
PHYSICAL = [
    *KEYS,
    "original_sequence_id",
    *IDENTITY,
    "source_timestamp_s",
    "available_timestamp_s",
    "calibration_id",
    "lever_arm_compensated",
    "sample_rate_hz",
    *LOW,
    *HIGH,
]
CLASSICAL = {
    "identity",
    "bias_mean",
    "affine",
    "ridge_affine",
    "learned_fir_causal",
    "arx_linear",
    "ridge_tuned",
    "fir_tuned",
    "arx_tuned",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(
        json.dumps(
            json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
    logger.info("WRITES {}", path)


def recorded_path(value: str, root: Path, project: Path) -> Path:
    """Relocate a copied run without altering any frozen source metadata."""
    normalized = str(value).replace("\\", "/")
    anchor = f"/{root.name}/"
    if anchor in "/" + normalized:
        suffix = ("/" + normalized).split(anchor, 1)[1]
        return root / suffix
    for anchor in ("src/virtual_imu/", "configs/", "data/"):
        if anchor in normalized:
            candidate = project / anchor / normalized.split(anchor, 1)[1]
            if candidate.exists():
                return candidate
    path = Path(value)
    return path if path.is_absolute() else project / path


def _checked_inventory(inventory: dict, root: Path, project: Path, checked: dict) -> None:
    if not inventory:
        raise ValueError("Required source/artifact checksum inventory is missing")
    for name, expected in inventory.items():
        path = recorded_path(name, root, project)
        key = str(path.resolve())
        if key in checked and checked[key] != expected:
            raise ValueError("Conflicting provenance hashes for one artifact")
        if key not in checked:
            if not path.is_file() or sha(path) != expected:
                raise ValueError(f"Source/artifact checksum differs: {path}")
            checked[key] = expected


def verify_complete_pipeline(state: dict, root: Path, project: Path) -> tuple[list[dict], dict]:
    """Gate counts first, then validate identities, selection ranks and all inventories."""
    cells = state.get("cells", {})
    finals = [
        job for job in state.get("jobs", {}).values() if job.get("phase", "").startswith("final_")
    ]
    if (
        state.get("status") != "complete"
        or state.get("smoke") is not False
        or len(cells) != 93
        or len(finals) != 558
        or any(
            job.get("status") != "complete" or job.get("evaluation_status") != "complete"
            for job in finals
        )
    ):
        raise ValueError(
            "Downstream access requires nonsmoke 93 selected cells and all 558 final evaluations complete"
        )
    if state.get("selection_uses_test") is not False or any(
        "selected" not in cell for cell in cells.values()
    ):
        raise ValueError("All 93 validation-only selections must be frozen")
    seeds, variants = state.get("final_seeds", []), state.get("final_variants", [])
    if len(seeds) != 3 or len(set(seeds)) != 3 or set(variants) != {"default", "tuned"}:
        raise ValueError("Require both final variants and three fixed seeds")
    expected = {
        (cell, f"final_{variant}", seed) for cell in cells for variant in variants for seed in seeds
    }
    actual = [(job["cell"], job["phase"], job["seed"]) for job in finals]
    if len(set(actual)) != 558 or set(actual) != expected:
        raise ValueError("Final matrix identities differ from all 93 x 2 x 3 cells")
    config = state["contract"]["config"]
    declared = {
        f"{p['id']}/{c['id']}"
        for p in config["protocols"]
        for c in config["candidates"]
        if p["task"] in c["tasks"]
    }
    if set(cells) != declared or set(state["contract"].get("cells", [])) != set(cells):
        raise ValueError("Declared protocol/family matrix differs from the frozen source contract")
    if yaml.safe_load((root / "matrix_config.yaml").read_text(encoding="utf-8")) != config:
        raise ValueError("Frozen HPO matrix YAML differs from pipeline contract")
    counts = state.get("counts", {})
    if any(
        counts.get(key) != value
        for key, value in {
            "cells": 93,
            "selected_cells": 93,
            "final_evaluated": 558,
            "expected_final_runs": 558,
        }.items()
    ):
        raise ValueError("Completed HPO count summary differs")
    checked = {}
    _checked_inventory(state["contract"].get("source_sha256", {}), root, project, checked)
    required_sources = [
        "eval/reviewer_evaluation.py",
        "eval/formal_evaluation.py",
        "eval/baselines.py",
        "eval/downstream_attitude.py",
        "training/reviewer_hpo.py",
    ]
    if any(
        str((project / "src/virtual_imu" / name).resolve()) not in checked
        for name in required_sources
    ):
        raise ValueError(
            "Frozen source inventory does not bind the executed evaluator and baseline implementation"
        )
    _checked_inventory(state.get("input_artifact_sha256", {}), root, project, checked)
    referenced = set()
    for identifier in cells:
        _verify_selection(identifier, state, root)
        referenced.update(cells[identifier]["selected"]["selection_jobs"])
    jobs = list(finals) + [state["jobs"][name] for name in sorted(referenced)]
    for job in jobs:
        directory = recorded_path(job["run_dir"], root, project)
        frozen = recorded_path(job["config"], root, project)
        _checked_inventory({str(frozen): job["config_sha256"]}, root, project, checked)
        training = {
            str(directory / name): value for name, value in job.get("artifact_sha256", {}).items()
        }
        _checked_inventory(training, root, project, checked)
        if not job["phase"].startswith("final_"):
            continue
        if (
            sha(directory / "best.ckpt") != job["checkpoint_sha256"]
            or sha(directory / "config.yaml") != job["run_config_sha256"]
        ):
            raise ValueError("Final checkpoint/config identity differs")
        run_config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        if run_config != yaml.safe_load(frozen.read_text(encoding="utf-8")):
            raise ValueError("Final run configuration differs from the frozen optimization job")
        report_path = root / "evaluations" / f"{job['id']}.json"
        _checked_inventory({str(report_path): job["evaluation_sha256"]}, root, project, checked)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("job_id") != job["id"]
            or report.get("checkpoint_sha256") != job["checkpoint_sha256"]
        ):
            raise ValueError("Final evaluation report is not bound to its frozen checkpoint")
        protocol_id, family = job["cell"].split("/", 1)
        rp = run_config.get("reviewer_protocol", {})
        metrics = report.get("metrics", {})
        if (
            rp.get("phase") != job["phase"]
            or rp.get("protocol_id") != protocol_id
            or rp.get("model_family") != family
            or rp.get("selection_scope") != "validation_only"
            or run_config.get("seed") != job["seed"]
            or metrics.get("task") != cells[job["cell"]]["protocol"]["task"]
            or metrics.get("reviewer_protocol") != rp
            or metrics.get("seed") != job["seed"]
        ):
            raise ValueError("Final evaluation task/fold/family/variant/seed provenance differs")
        inventory = job.get("evaluation_artifact_sha256", {})
        _checked_inventory(inventory, root, project, checked)
        bound = {str(recorded_path(name, root, project).resolve()) for name in inventory}
        required = {
            str((directory / "evaluation" / name).resolve())
            for name in (
                "test_predictions.parquet",
                "baseline_train_state.json",
                "provenance.json",
                "metrics.json",
            )
        }
        if not required <= bound:
            raise ValueError(
                "Final evaluation inventory omits required test/baseline/provenance artifacts"
            )
        # Reject additional or replaced evaluation payloads that were not inventoried.
        actual_files = {
            str(path.resolve())
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in {".json", ".csv", ".parquet"}
        }
        if not actual_files <= bound:
            raise ValueError("Final evaluation has uninventoried payload files")
    return finals, {
        "verified_files": checked,
        "selected_cells": 93,
        "final_evaluations": 558,
        "selection_uses_test": False,
    }


def select_penn_baseline(state: dict) -> dict:
    if state.get("fit_split") != "train" or state.get("selection_split") != "validation":
        raise ValueError("Classical replay must use saved train-fit, validation-selected states")
    pairs = [pair for pair in state.get("pairs", []) if pair.get("identity") == PENN_IDENTITY]
    if len(pairs) != 1:
        raise ValueError("Need exactly one calibrated Penn sensor/reference baseline mapping")
    if (
        pairs[0].get("fixed", {}).get("identity") != PENN_IDENTITY
        or pairs[0]["fixed"].get("fit_split") != "train"
    ):
        raise ValueError("Fixed classical state must bind the same Penn mapping and training split")
    return pairs[0]


def validate_stream(prediction: pd.DataFrame, canonical: pd.DataFrame, recording: str) -> dict:
    if (
        prediction.empty
        or prediction.duplicated(KEYS).any()
        or set(prediction.original_sequence_id.astype(str)) != {recording}
    ):
        raise ValueError("Prediction recording identity/unique physical support differs")
    if not all(prediction[key].eq(value).all() for key, value in PENN_IDENTITY.items()):
        raise ValueError("Wrong calibrated Penn sensor/reference mapping")
    if (
        not prediction.calibration_id.eq("penncosyvio_published_body_extrinsics_v1").all()
        or not prediction.lever_arm_compensated.eq(True).all()
    ):
        raise ValueError("Predictions must already be in the calibrated rig body frame/origin")
    if (
        not prediction.sample_rate_hz.eq(40.0).all()
        or not np.isfinite(prediction[[*LOW, *HIGH, *VIRTUAL]].to_numpy(dtype=float)).all()
    ):
        raise ValueError("Require finite six-axis predictions and 40Hz physical support")
    pieces = []
    for _, group in canonical.groupby("sequence_id", sort=False):
        group = group.sort_values("timestamp_s")
        if len(group) < 320:
            continue
        count = ((len(group) - 320) // 40) * 40 + 320
        pieces.append(group.iloc[:count])
    if not pieces:
        raise ValueError("No predeclared >=320-point canonical support")
    expected = pd.concat(pieces, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    actual = prediction.sort_values(KEYS).reset_index(drop=True)
    if len(actual) != len(expected) or any(column not in actual for column in PHYSICAL):
        raise ValueError("Prediction support differs from the canonical 320/stride40 policy")
    try:
        pd.testing.assert_frame_equal(
            actual[PHYSICAL], expected[PHYSICAL], check_dtype=False, check_exact=True
        )
    except AssertionError as error:
        raise ValueError(
            "Prediction physical rows/clock/measurements differ from canonical support"
        ) from error
    for _, group in actual.groupby("sequence_id", sort=False):
        if not np.allclose(np.diff(group.timestamp_s), 0.025, atol=1e-6, rtol=0):
            raise ValueError("Predictions cross a physical-time gap")
    origin = source_clock_origin_s(actual)
    if (
        not np.isfinite(actual.available_timestamp_s).all()
        or (actual.available_timestamp_s < actual.timestamp_s - 1e-6).any()
    ):
        raise ValueError("Availability timing cannot precede physical sample time")
    if all(f"model_{axis}" in actual for axis in AXES) and not np.array_equal(
        actual[[f"model_{axis}" for axis in AXES]], actual[VIRTUAL]
    ):
        raise ValueError("Stored model output differs from the virtual six-axis prediction")
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(actual[PHYSICAL], index=False).to_numpy().tobytes()
    ).hexdigest()
    return {
        "rows": len(actual),
        "physical_sha256": fingerprint,
        "source_clock_origin_s": origin,
        "recording": recording,
    }


def verify_completed(directory: Path, contract_sha: str) -> dict:
    path = directory / "artifact_manifest.json"
    if not path.is_file():
        raise ValueError(f"Result cache is not complete: {directory}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("contract_sha256") != contract_sha:
        raise ValueError("Completed result cache has a different contract")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    if actual != set(manifest["files"]):
        raise ValueError("Completed result cache file inventory differs")
    for name, expected in manifest["files"].items():
        if sha(directory / name) != expected:
            raise ValueError(f"Completed result checksum differs: {directory / name}")
    return manifest


def finish_record(staging: Path, destination: Path, contract_sha: str) -> None:
    if destination.exists():
        raise FileExistsError(f"Refuse to overwrite completed result: {destination}")
    (staging / "PARTIAL.json").unlink(missing_ok=True)
    files = {
        path.relative_to(staging).as_posix(): sha(path)
        for path in staging.rglob("*")
        if path.is_file()
    }
    write_once(
        staging / "artifact_manifest.json",
        {"status": "complete", "contract_sha256": contract_sha, "files": files},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)


def _historical_helpers():
    path = Path(__file__).with_name("evaluate_downstream_benchmark.py")
    spec = importlib.util.spec_from_file_location("historical_downstream_helpers", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _canonical_recording(recording, config, dataset_manifest, root, project):
    paths = sorted(Path(config["paired_root"]).glob(f"penncosyvio__{recording}__seg*.parquet"))
    if not paths:
        raise ValueError("Canonical Penn recording is unavailable")
    declared = {
        row["sequence_id"]: row
        for row in dataset_manifest["records"]
        if row.get("dataset_id") == "penncosyvio"
    }
    frames, provenance = [], []
    for path in paths:
        frame = pd.read_parquet(path)
        identifiers = set(frame.sequence_id.astype(str))
        if len(identifiers) != 1 or next(iter(identifiers)) not in declared:
            raise ValueError("Canonical Penn sequence is absent from the frozen dataset manifest")
        expected = declared[next(iter(identifiers))]["sha256"]
        if sha(path) != expected:
            raise ValueError("Canonical Penn sequence checksum changed")
        frames.append(frame)
        provenance.append({"path": str(path.resolve()), "sha256": expected})
    return pd.concat(frames, ignore_index=True), provenance


def _job_metadata(job, state, root, project):
    directory = recorded_path(job["run_dir"], root, project)
    config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    protocol, family = job["cell"].split("/", 1)
    actual = config["reviewer_protocol"]
    variant = job["phase"].removeprefix("final_")
    expected_parameters = (
        state["cells"][job["cell"]]["trials"][0]["parameters"]
        if variant == "default"
        else state["cells"][job["cell"]]["selected"]["trial"]["parameters"]
    )
    if (
        actual.get("phase") != job["phase"]
        or actual.get("variant") != variant
        or actual.get("protocol_id") != protocol
        or actual.get("model_family") != family
        or actual.get("selection_scope") != "validation_only"
        or actual.get("parameters") != expected_parameters
        or config["seed"] != job["seed"]
    ):
        raise ValueError("Final job is not the matched frozen family/variant/seed/selection")
    if (
        config["data"].get("feature_columns") != LOW
        or config["model"].get("in_channels") != 6
        or config["model"].get("out_channels") != 6
    ):
        raise ValueError("Downstream requires the actual six-channel paired correction task")
    provenance = json.loads((directory / "evaluation/provenance.json").read_text(encoding="utf-8"))
    if (
        provenance.get("test_used_for_selection") is not False
        or provenance["checkpoint"]["sha256"] != job["checkpoint_sha256"]
        or provenance["config"]["sha256"] != job["run_config_sha256"]
    ):
        raise ValueError("Prediction provenance lacks its frozen final checkpoint/config identity")
    return directory, config


def process_unit(protocol, recording, jobs, state, config, root, project, staging):
    historical = _historical_helpers()
    first_dir, _first_config = _job_metadata(jobs[0], state, root, project)
    dataset_path = first_dir / "data_manifest.json"
    if sha(dataset_path) != state["dataset_manifest_sha256"]:
        raise ValueError("Canonical data manifest differs from frozen HPO dataset")
    canonical, source_records = _canonical_recording(
        recording, config, json.loads(dataset_path.read_text(encoding="utf-8")), root, project
    )
    common, support, baseline, baseline_sha = None, None, None, None
    extra, mapping, records = (
        {},
        {
            "high": {
                "method": "HIGH",
                "seed": None,
                "category": "measured_reference",
                "family": None,
                "variant": None,
            }
        },
        [],
    )
    for index, job in enumerate(
        sorted(jobs, key=lambda row: (row["cell"], row["phase"], row["seed"]))
    ):
        directory, run_config = _job_metadata(job, state, root, project)
        path = directory / "evaluation/test_predictions.parquet"
        logger.info("READS frozen final predictions {}", path)
        frame = pd.read_parquet(path)
        group = frame[frame.dataset_id.eq("penncosyvio")].sort_values(KEYS).reset_index(drop=True)
        group, recovery = historical.recover_source_clock(group, Path(config["paired_root"]))
        actual_support = validate_stream(group, canonical, recording)
        if support is not None and actual_support != support:
            raise ValueError("Models/variants/seeds do not share identical physical test support")
        support = actual_support
        saved_state = json.loads(
            (directory / "evaluation/baseline_train_state.json").read_text(encoding="utf-8")
        )
        selected = select_penn_baseline(saved_state)
        selected_sha = canonical_sha(selected)
        if baseline_sha is not None and selected_sha != baseline_sha:
            raise ValueError(
                "Classical baseline state differs across family/variant/seed; cannot deduplicate"
            )
        baseline, baseline_sha = selected, selected_sha
        replay = _predict_paired_classical(selected, group)
        if set(replay) != CLASSICAL:
            raise ValueError(
                "Saved reviewer classical methods differ from the declared nine methods"
            )
        for name, values in replay.items():
            columns = [f"{name}_{axis}" for axis in AXES]
            if any(column not in group for column in columns) or not np.allclose(
                group[columns], values, rtol=1e-10, atol=1e-12
            ):
                raise ValueError(
                    "Saved classical predictions do not match train/validation-only state replay"
                )
        if common is None:
            common = group[PHYSICAL].copy()
        prefix = f"neural_{index:03d}"
        extra.update(
            {
                f"{axis}_{prefix}": group[column].to_numpy(dtype=float)
                for axis, column in zip(AXES, VIRTUAL, strict=True)
            }
        )
        family = job["cell"].split("/", 1)[1]
        variant = job["phase"].removeprefix("final_")
        mapping[prefix] = {
            "method": f"{family}/{variant}",
            "seed": job["seed"],
            "category": "neural",
            "family": family,
            "variant": variant,
        }
        records.append(
            {
                "job_id": job["id"],
                "cell": job["cell"],
                "seed": job["seed"],
                "variant": variant,
                "prediction_path": str(path),
                "prediction_sha256": sha(path),
                "checkpoint_sha256": job["checkpoint_sha256"],
                "config_sha256": job["run_config_sha256"],
                "evaluation_provenance_sha256": sha(directory / "evaluation/provenance.json"),
                "baseline_state_sha256": sha(directory / "evaluation/baseline_train_state.json"),
                "penn_baseline_content_sha256": selected_sha,
                "source_clock_recovery": recovery,
                "physical_support": support,
                "input_window_samples": run_config["data"]["window_size"],
                "model_config": run_config["model"],
            }
        )
    if common is None or baseline is None:
        raise ValueError("No complete model streams")
    for name, values in _predict_paired_classical(baseline, common).items():
        prefix = f"classic_{name}"
        extra.update(
            {f"{axis}_{prefix}": values[:, axis_index] for axis_index, axis in enumerate(AXES)}
        )
        mapping[prefix] = {
            "method": name,
            "seed": None,
            "category": "classical",
            "family": None,
            "variant": "validation_tuned" if name.endswith("_tuned") else "fixed",
        }
    combined = pd.concat([common, pd.DataFrame(extra, index=common.index)], axis=1)
    gt_path = Path(config["output_root"]) / f"{recording}.parquet"
    logger.info("READS independent optical GT {}; no fitted lag/rotation/initialization", gt_path)
    truth = pd.read_parquet(gt_path)
    args = {
        "prefixes": list(mapping),
        "window_samples": config["protocol"]["window_samples"],
        "rate_hz": 40.0,
        "max_gt_gap_s": config["max_gt_gap_s"],
        "gt_timestamp_origin_s": support["source_clock_origin_s"],
    }
    results = {
        "attitude": evaluate_attitude_stream(combined, truth, **args),
        "strapdown": evaluate_strapdown_stream(
            combined, truth, gravity_world_mps2=config["protocol"]["gravity_world_mps2"], **args
        ),
    }
    tables = {"trials": [], "window_details": [], "full_segments": []}
    for task, result in results.items():
        for section, key in (
            ("trials", "windows"),
            ("window_details", "window_details"),
            ("full_segments", "full_recordings"),
        ):
            for row in result[key]:
                tables[section].append(
                    {
                        **row,
                        **mapping[row["method"]],
                        "task": task,
                        "protocol": protocol,
                        "recording": recording,
                    }
                )
    trials = pd.DataFrame(tables["trials"])
    summary = historical.aggregate_trials(
        trials.drop(columns=["protocol", "recording", "family", "variant", "category"])
    )
    summary["protocol"], summary["recording"] = protocol, recording
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(staging / f"{name}.csv", index=False)
        write_once(staging / f"{name}.json", rows)
    summary.to_csv(staging / "summary.csv", index=False)
    write_once(staging / "summary.json", summary.to_dict(orient="records"))
    combined.to_parquet(staging / "evaluated_signals.parquet", index=False)
    metadata = {
        "protocol": protocol,
        "recording": recording,
        "view": "separate_same_recording_pooled_reference"
        if protocol == "pooled"
        else "outer_recording_holdout",
        "physical_support": support,
        "methods": mapping,
        "neural_optimizer_runs": len(jobs),
        "neural_method_variants": len(
            {item["method"] for item in mapping.values() if item["category"] == "neural"}
        ),
        "unique_classical_methods": len(CLASSICAL),
        "method_variants_total": len({item["method"] for item in mapping.values()}),
        "stream_count": len(mapping),
        "baseline_deduplication": "one identical saved train/validation-only Penn state across all family/variant/seed jobs; one evaluation per classical method",
        "penn_baseline_content_sha256": baseline_sha,
        "canonical_sources": source_records,
        "independent_gt_path": str(gt_path),
        "independent_gt_sha256": sha(gt_path),
        "model_records": records,
        "support": {task: result["support"] for task, result in results.items()},
        "propagation_protocol": {task: result["protocol"] for task, result in results.items()},
        "interpretation": "offline stored predictions, oracle initial state, no test-selected method/seed, no deployable VIO or real-time latency claim; optimizer SD is not recording-level CI",
    }
    write_once(staging / "metrics.json", metadata)
    return metadata


def _gt_inventory(config, root, project):
    path = Path(config["output_root"]) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("source_revision") != config["source_revision"]:
        raise ValueError("Optical GT revision differs from the frozen downstream configuration")
    rows = {record["sequence"]: record for record in manifest["records"]}
    if set(rows) != {"af", "as", "bf", "bs"}:
        raise ValueError("Independent optical GT must retain all four recordings")
    checked = {str(path.resolve()): sha(path)}
    for row in rows.values():
        _checked_inventory(
            {
                row["path"]: row["sha256"],
                row["processed_path"]: row["processed_sha256"],
                str(Path(config["output_root"]) / f"{row['sequence']}.parquet"): row[
                    "processed_sha256"
                ],
            },
            root,
            project,
            checked,
        )
    return checked


def validate_protocol(config: dict) -> None:
    expected = {
        "rate_hz": 40.0,
        "window_samples": [80, 200, 400, 800],
        "endpoint_spans_s": [1.975, 4.975, 9.975, 19.975],
        "gravity_world_mps2": [0.0, 0.0, -9.80665],
        "initial_velocity": "fixed_past_0.1_second_optical_position_difference_once",
        "initial_velocity_must_use_past_only": True,
        "gyro_integration": "midpoint_SO3_exponential",
        "accel_integration": "trapezoid_world_frame",
        "gt_in_propagation": "forbidden",
        "method_selection_on_downstream_test": "forbidden",
        "gt_time_mapping": "source_timestamp_s_equals_timestamp_s_plus_recorded_source_origin",
    }
    if config.get("max_gt_gap_s") != 0.1 or any(
        config.get("protocol", {}).get(key) != value for key, value in expected.items()
    ):
        raise ValueError(
            "Frozen downstream protocol disagrees with the implemented propagation/initialization"
        )


def run(
    pipeline_root: Path,
    config_path: Path,
    out: Path,
    *,
    resume: bool = False,
    project: Path = PROJECT,
):
    root, project = pipeline_root.resolve(), project.resolve()
    logger.info(
        "READS reviewer final pipeline {}; SKIPS all searches, fitting and model inference", root
    )
    state = json.loads((root / "pipeline.json").read_text(encoding="utf-8"))
    finals, gate = verify_complete_pipeline(state, root, project)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_protocol(config)
    config = {
        **config,
        "paired_root": str(recorded_path(config["paired_root"], root, project)),
        "output_root": str(recorded_path(config["output_root"], root, project)),
    }
    families = {
        candidate["id"]
        for candidate in state["contract"]["config"]["candidates"]
        if "paired" in candidate["tasks"]
    }
    if len(families) != 10:
        raise ValueError("Require all ten declared paired model families")
    units = {}
    for protocol, recording in UNITS.items():
        selected = [job for job in finals if job["cell"].split("/", 1)[0] == protocol]
        expected = {
            (f"{protocol}/{family}", f"final_{variant}", seed)
            for family in families
            for variant in ("default", "tuned")
            for seed in state["final_seeds"]
        }
        if (
            len(selected) != 60
            or {(job["cell"], job["phase"], job["seed"]) for job in selected} != expected
        ):
            raise ValueError("Penn downstream unit is missing a declared family/variant/seed")
        units[protocol] = (recording, selected)
    source_paths = [
        Path(__file__),
        Path(__file__).with_name("evaluate_downstream_benchmark.py"),
        project / "src/virtual_imu/eval/downstream_attitude.py",
    ]
    contract = {
        "schema_version": 1,
        "pipeline_path": str(root),
        "pipeline_sha256": sha(root / "pipeline.json"),
        "downstream_config_sha256": sha(config_path),
        "gate": gate,
        "independent_gt_inventory": _gt_inventory(config, root, project),
        "postprocessor_sha256": {str(path.resolve()): sha(path) for path in source_paths},
        "units": {
            protocol: {"recording": recording, "job_ids": sorted(job["id"] for job in jobs)}
            for protocol, (recording, jobs) in units.items()
        },
        "expected_neural_streams_per_unit": 60,
        "expected_classical_methods_per_unit": 9,
        "expected_method_variants_per_unit": 30,
        "scientific_scope": "four outer Penn recordings; pooled bs separate; all methods and all three seeds retained; fixed oracle initialization; no downstream test selection",
    }
    contract_sha = canonical_sha(contract)
    if out.exists() and any(out.iterdir()):
        if (
            not resume
            or not (out / "contract.json").is_file()
            or json.loads((out / "contract.json").read_text(encoding="utf-8")) != contract
        ):
            raise ValueError(
                "Existing downstream output requires --resume and the identical frozen contract; no overwrite"
            )
        logger.info("SKIPS verified frozen downstream contract {}", contract_sha)
        if sha(out / "config_snapshot.yaml") != contract["downstream_config_sha256"]:
            raise ValueError("Cached frozen downstream config snapshot checksum differs")
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_once(out / "contract.json", contract)
        (out / "config_snapshot.yaml").write_bytes(config_path.read_bytes())
        write_once(
            out / "PARTIAL.json",
            {
                "created_utc": datetime.now(UTC).isoformat(),
                "contract_sha256": contract_sha,
                "status": "incomplete_until_complete_json_exists",
            },
        )
    logger.add(out / "postprocessor.log", encoding="utf-8")
    print(f"FROZEN_REVIEWER_DOWNSTREAM_CONTRACT_SHA256={contract_sha}", flush=True)
    for protocol, (recording, jobs) in units.items():
        destination = out / "units" / protocol
        if destination.exists():
            verify_completed(destination, contract_sha)
            logger.info("SKIPS hash-verified completed downstream unit {}", protocol)
            continue
        partial_root = out / ".partial"
        partial_root.mkdir(exist_ok=True)
        staging = (
            partial_root
            / f"{protocol}__attempt_{len(list(partial_root.glob(protocol + '__attempt_*'))):04d}"
        )
        staging.mkdir(exist_ok=False)
        write_once(
            staging / "PARTIAL.json",
            {
                "protocol": protocol,
                "contract_sha256": contract_sha,
                "status": "partial_attempt_preserved_on_failure",
            },
        )
        process_unit(protocol, recording, jobs, state, config, root, project, staging)
        finish_record(staging, destination, contract_sha)
    overview = out / "overview"
    if overview.exists():
        verify_completed(overview, contract_sha)
    else:
        staging = (
            out
            / ".partial"
            / f"overview__attempt_{len(list((out / '.partial').glob('overview__attempt_*'))):04d}"
        )
        staging.mkdir()
        write_once(
            staging / "PARTIAL.json",
            {"contract_sha256": contract_sha, "status": "partial_aggregate_attempt"},
        )
        summaries, unit_records = [], []
        for protocol in units:
            directory = out / "units" / protocol
            verify_completed(directory, contract_sha)
            summaries.append(pd.read_csv(directory / "summary.csv"))
            unit_records.append(
                json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            )
        pd.concat(summaries, ignore_index=True).to_csv(
            staging / "all_fold_summary.csv", index=False
        )
        write_once(
            staging / "metrics.json",
            {
                "status": "complete",
                "contract_sha256": contract_sha,
                "units": [
                    {
                        key: row[key]
                        for key in (
                            "protocol",
                            "recording",
                            "view",
                            "physical_support",
                            "neural_optimizer_runs",
                            "unique_classical_methods",
                            "method_variants_total",
                            "stream_count",
                        )
                    }
                    for row in unit_records
                ],
                "primary_outer_recordings": ["af", "as", "bf", "bs"],
                "pooled_bs": "separate same-recording comparison; never a fifth independent recording",
                "optimizer_SD": "optimization variation only, not window or recording confidence intervals",
            },
        )
        finish_record(staging, overview, contract_sha)
    completion = {
        "status": "complete",
        "contract_sha256": contract_sha,
        "unit_manifests": {
            protocol: sha(out / "units" / protocol / "artifact_manifest.json") for protocol in units
        },
        "overview_manifest_sha256": sha(overview / "artifact_manifest.json"),
    }
    if (out / "complete.json").exists():
        if json.loads((out / "complete.json").read_text(encoding="utf-8")) != completion:
            raise ValueError("Completed top-level cache identity differs")
    else:
        write_once(out / "complete.json", completion)
    logger.info(
        "COMPLETED five downstream views, four independent outer recordings; SKIPS ranking/test selection"
    )
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/downstream_penn.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.pipeline_root, args.config, args.out, resume=args.resume)
    print(
        json.dumps(
            {
                "status": result["status"],
                "contract_sha256": result["contract_sha256"],
                "out": str(args.out),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
