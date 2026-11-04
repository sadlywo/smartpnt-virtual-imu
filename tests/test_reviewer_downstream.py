"""Final-only downstream gating, scientific support and immutable resume tests."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_reviewer_downstream.py"
    spec = importlib.util.spec_from_file_location("reviewer_downstream", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def stream(rows=360, recording="bs"):
    from virtual_imu.eval.formal_evaluation import HIGH, LOW, VIRTUAL

    frame = pd.DataFrame({"timestamp_s": 0.2 + np.arange(rows) / 40})
    frame["source_timestamp_s"] = frame.timestamp_s + 0.005
    frame["available_timestamp_s"] = frame.timestamp_s + 0.9
    frame["sequence_id"] = f"penncosyvio__{recording}__seg0000"
    frame["original_sequence_id"] = recording
    frame["dataset_id"] = "penncosyvio"
    frame["low_sensor"], frame["high_sensor"] = "Tango Top IMU", "ADIS16488"
    frame["calibration_id"] = "penncosyvio_published_body_extrinsics_v1"
    frame["lever_arm_compensated"] = True
    frame["sample_rate_hz"] = 40.0
    values = np.tile([0, 0, 0, 0, 0, 9.80665], (rows, 1))
    frame[LOW], frame[HIGH], frame[VIRTUAL] = values, values, values
    return frame


def test_incomplete_hpo_gate_fails_before_any_prediction_access(tmp_path):
    downstream = module()
    state = {"status": "searching", "smoke": False, "cells": {}, "jobs": {}}
    with pytest.raises(ValueError, match=r"93|558"):
        downstream.verify_complete_pipeline(state, tmp_path, tmp_path)


def test_canonical_support_and_rig_frame_are_enforced():
    downstream = module()
    canonical = stream()
    prediction = canonical.iloc[:360].copy()
    assert downstream.validate_stream(prediction, canonical, "bs")["rows"] == 360
    with pytest.raises(ValueError, match="support"):
        downstream.validate_stream(prediction.iloc[:-1], canonical, "bs")
    prediction.loc[0, "calibration_id"] = "ADIS_axes_without_rig_rotation"
    with pytest.raises(ValueError, match="rig"):
        downstream.validate_stream(prediction, canonical, "bs")


def test_availability_time_cannot_replace_recorded_physical_source_clock():
    downstream = module()
    canonical = stream()
    prediction = canonical.copy()
    prediction["source_timestamp_s"] = prediction.available_timestamp_s
    with pytest.raises(ValueError, match=r"physical|clock"):
        downstream.validate_stream(prediction, canonical, "bs")


def test_completed_cache_rejects_tampering_and_partial_is_not_complete(tmp_path):
    downstream = module()
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "PARTIAL.json").write_text("{}")
    with pytest.raises(ValueError, match="complete"):
        downstream.verify_completed(partial, "contract")
    (partial / "result.json").write_text("{}")
    downstream.finish_record(partial, tmp_path / "done", "contract")
    downstream.verify_completed(tmp_path / "done", "contract")
    (tmp_path / "done/result.json").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        downstream.verify_completed(tmp_path / "done", "contract")


def test_baseline_deduplication_checks_full_saved_state():
    downstream = module()
    state = {
        "fit_split": "train",
        "selection_split": "validation",
        "pairs": [
            {
                "identity": downstream.PENN_IDENTITY,
                "fixed": {
                    "bias": [0] * 6,
                    "identity": downstream.PENN_IDENTITY,
                    "fit_split": "train",
                },
                "ridge_tuned": {"weights": [[1]]},
            }
        ],
    }
    first = downstream.select_penn_baseline(state)
    changed = json.loads(json.dumps(state))
    changed["pairs"][0]["ridge_tuned"]["weights"][0][0] = 2
    assert downstream.canonical_sha(first) != downstream.canonical_sha(
        downstream.select_penn_baseline(changed)
    )


def test_declared_protocol_cannot_disagree_with_actual_propagation():
    import yaml

    downstream = module()
    config = yaml.safe_load(
        (downstream.PROJECT / "configs/downstream_penn.yaml").read_text(encoding="utf-8")
    )
    config["protocol"]["rate_hz"] = 41.0
    with pytest.raises(ValueError, match="protocol"):
        downstream.validate_protocol(config)


def test_actually_read_gt_file_must_match_manifest_even_when_original_copy_exists(tmp_path):
    downstream = module()
    original, evaluated = tmp_path / "original_gt", tmp_path / "evaluated_gt"
    original.mkdir()
    evaluated.mkdir()
    rows = []
    for recording in ("af", "as", "bf", "bs"):
        path = original / f"{recording}.parquet"
        path.write_bytes(b"official fixture bytes")
        raw = original / f"{recording}.txt"
        raw.write_bytes(b"raw fixture bytes")
        (evaluated / path.name).write_bytes(path.read_bytes())
        rows.append(
            {
                "sequence": recording,
                "path": str(raw),
                "sha256": downstream.sha(raw),
                "processed_path": str(path),
                "processed_sha256": downstream.sha(path),
            }
        )
    (evaluated / "manifest.json").write_text(
        json.dumps({"source_revision": "pinned", "records": rows}), encoding="utf-8"
    )
    (evaluated / "af.parquet").write_bytes(b"changed evaluated GT")
    with pytest.raises(ValueError, match="checksum"):
        downstream._gt_inventory(
            {"output_root": str(evaluated), "source_revision": "pinned"},
            tmp_path / "pipeline",
            tmp_path,
        )


def _complete_synthetic_pipeline(tmp_path):
    """Small real files, full production gate; no gate or propagation mocks."""
    import copy

    import yaml

    from virtual_imu.eval.reviewer_evaluation import _predict_paired_classical

    downstream = module()
    root = tmp_path / "synthetic_pipeline"
    root.mkdir()
    for name in ("selections", "evaluations", "configs", "experiments"):
        (root / name).mkdir()
    paired_root, gt_root, raw_root = [
        tmp_path / name for name in ("canonical_paired", "independent_optical", "raw_optical")
    ]
    for path in (paired_root, gt_root, raw_root):
        path.mkdir()
    fixed = {
        "identity": downstream.PENN_IDENTITY,
        "fit_split": "train",
        "fit_samples": 360,
        "bias": [0.0] * 6,
        "affine": np.vstack([np.eye(6), np.zeros(6)]).tolist(),
        "ridge_affine": np.vstack([np.eye(6), np.zeros(6)]).tolist(),
        "system_id": {
            "fir_taps": 1,
            "arx_taps": 1,
            "fir_weights": np.eye(6).tolist(),
            "fir_bias": [0.0] * 6,
            "arx_weights": np.vstack([np.eye(6), np.zeros((6, 6))]).tolist(),
            "arx_bias": [0.0] * 6,
        },
    }
    selected_pair = {
        "identity": downstream.PENN_IDENTITY,
        "fixed": fixed,
        "ridge_tuned": {"weights": np.eye(6).tolist(), "bias": [0.0] * 6},
        "fir_tuned": {
            "weights": np.eye(6).tolist(),
            "bias": [0.0] * 6,
            "taps": 1,
            "include_delta": False,
        },
        "arx_tuned": {
            "weights": np.vstack([np.eye(6), np.zeros((6, 6))]).tolist(),
            "bias": [0.0] * 6,
            "taps": 1,
            "include_delta": True,
        },
    }
    baseline = {"fit_split": "train", "selection_split": "validation", "pairs": [selected_pair]}

    def dump(path, value):
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    records, gt_records, prediction_paths = [], [], {}
    for recording in ("af", "as", "bf", "bs"):
        frame = stream(recording=recording)
        canonical_path = paired_root / f"penncosyvio__{recording}__seg0000.parquet"
        frame.to_parquet(canonical_path, index=False)
        records.append(
            {
                "dataset_id": "penncosyvio",
                "sequence_id": frame.sequence_id.iloc[0],
                "sha256": downstream.sha(canonical_path),
                "path": str(canonical_path),
            }
        )
        for method, values in {
            "model": frame[downstream.VIRTUAL].to_numpy(),
            **_predict_paired_classical(selected_pair, frame),
        }.items():
            for index, axis in enumerate(downstream.AXES):
                frame[f"{method}_{axis}"] = values[:, index]
        prediction_path = tmp_path / f"prediction_{recording}.parquet"
        frame.to_parquet(prediction_path, index=False)
        prediction_paths[recording] = prediction_path
        times = np.arange(400) / 40
        truth = pd.DataFrame({"timestamp_s": times})
        for i in range(3):
            for j in range(3):
                truth[f"r{i}{j}"] = float(i == j)
        truth[["px_m", "py_m", "pz_m"]] = 0.0
        gt_path = gt_root / f"{recording}.parquet"
        truth.to_parquet(gt_path, index=False)
        raw_path = raw_root / f"{recording}.txt"
        transforms = np.tile(
            np.column_stack([np.eye(3), np.zeros(3)]).reshape(1, 12), (len(times), 1)
        )
        np.savetxt(raw_path, np.column_stack([times, transforms]))
        gt_records.append(
            {
                "sequence": recording,
                "path": str(raw_path),
                "sha256": downstream.sha(raw_path),
                "processed_path": str(gt_path),
                "processed_sha256": downstream.sha(gt_path),
            }
        )
    dump(
        gt_root / "manifest.json",
        {"source_revision": "0" * 40, "records": gt_records, "synthetic_fixture": True},
    )
    manifest = tmp_path / "canonical_manifest.json"
    dump(manifest, {"records": records, "synthetic_fixture": True})
    config = yaml.safe_load(
        (downstream.PROJECT / "configs/downstream_penn.yaml").read_text(encoding="utf-8")
    )
    config.update(source_revision="0" * 40, paired_root=str(paired_root), output_root=str(gt_root))
    downstream_config = tmp_path / "downstream_fixture.yaml"
    downstream_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    families = [
        "tcn",
        "plain_cnn",
        "gru",
        "cnn_bilstm",
        "transformer",
        "informer",
        "patchtst",
        "itransformer",
        "deepgyro_adapter",
        "tlio_resnet18_adapter",
    ]
    protocols = [
        {"id": name, "task": "paired"}
        for name in [
            "pooled",
            "alimu_only",
            "penn_only",
            "penn_loso_af",
            "penn_loso_as",
            "penn_loso_bf",
        ]
    ] + [{"id": f"navigation_rotation_{i}", "task": "navigation"} for i in range(3)]
    candidates = [{"id": name, "tasks": ["paired", "navigation"]} for name in families] + [
        {"id": "tartanimu_window_adapter", "tasks": ["navigation"]}
    ]
    matrix_config = {
        "protocols": protocols,
        "candidates": candidates,
        "seed": 100,
        "confirmation_seed": 101,
        "search": {"rung_epochs": [1], "rung_keep": [1], "trials": 1},
        "synthetic_fixture": True,
    }
    (root / "matrix_config.yaml").write_text(yaml.safe_dump(matrix_config), encoding="utf-8")
    state = {
        "schema_version": 1,
        "status": "complete",
        "smoke": False,
        "selection_uses_test": False,
        "dataset_manifest_sha256": downstream.sha(manifest),
        "final_seeds": [1, 2, 3],
        "final_variants": ["default", "tuned"],
        "cells": {},
        "jobs": {},
        "counts": {
            "cells": 93,
            "selected_cells": 93,
            "final_evaluated": 558,
            "expected_final_runs": 558,
        },
        "input_artifact_sha256": {str(manifest): downstream.sha(manifest)},
    }

    def job(cell, phase, seed):
        protocol, family = cell.split("/")
        identifier = f"{protocol}__{family}__{phase}__s{seed}"
        directory = root / "experiments" / identifier
        directory.mkdir()
        task = state["cells"][cell]["protocol"]["task"]
        run_config = {
            "seed": seed,
            "data": {"feature_columns": downstream.LOW, "window_size": 80},
            "model": {
                "type": family,
                "in_channels": 6,
                "out_channels": 6 if task == "paired" else 3,
                "target_mode": "residual",
            },
            "reviewer_protocol": {
                "protocol_id": protocol,
                "model_family": family,
                "phase": phase,
                "variant": phase.removeprefix("final_") if phase.startswith("final_") else "search",
                "selection_scope": "validation_only",
                "parameters": {},
            },
        }
        text = yaml.safe_dump(run_config)
        frozen = root / "configs" / f"{identifier}.yaml"
        frozen.write_text(text, encoding="utf-8")
        (directory / "config.yaml").write_text(text, encoding="utf-8")
        (directory / "best.ckpt").write_bytes(
            b"synthetic metadata fixture, not a trained checkpoint"
        )
        (directory / "data_manifest.json").write_bytes(manifest.read_bytes())
        dump(directory / "provenance.json", {"synthetic_fixture": True})
        row = {
            "id": identifier,
            "cell": cell,
            "phase": phase,
            "trial_id": "trial_000",
            "rung": 0 if phase == "search" else None,
            "seed": seed,
            "epochs": 1,
            "status": "complete",
            "score": 1.0,
            "config": str(frozen),
            "config_sha256": downstream.sha(frozen),
            "run_dir": str(directory),
            "checkpoint_sha256": downstream.sha(directory / "best.ckpt"),
            "run_config_sha256": downstream.sha(directory / "config.yaml"),
            "artifact_sha256": {
                name: downstream.sha(directory / name)
                for name in ["best.ckpt", "config.yaml", "data_manifest.json", "provenance.json"]
            },
        }
        if phase.startswith("final_"):
            evaluation = directory / "evaluation"
            evaluation.mkdir()
            recording = downstream.UNITS.get(protocol, "bs")
            os.link(prediction_paths[recording], evaluation / "test_predictions.parquet")
            dump(evaluation / "baseline_train_state.json", baseline)
            dump(
                evaluation / "provenance.json",
                {
                    "test_used_for_selection": False,
                    "checkpoint": {"sha256": row["checkpoint_sha256"]},
                    "config": {"sha256": row["run_config_sha256"]},
                },
            )
            metrics = {
                "task": task,
                "reviewer_protocol": run_config["reviewer_protocol"],
                "seed": seed,
                "synthetic_fixture": True,
            }
            dump(evaluation / "metrics.json", metrics)
            report = root / "evaluations" / f"{identifier}.json"
            dump(
                report,
                {
                    "job_id": identifier,
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "metrics": metrics,
                },
            )
            row.update(
                evaluation_status="complete",
                evaluation_sha256=downstream.sha(report),
                evaluation_artifact_sha256={
                    str(path): downstream.sha(path)
                    for path in directory.rglob("*")
                    if path.is_file() and path.suffix in {".json", ".csv", ".parquet"}
                },
            )
            row["evaluation_artifact_sha256"][str(report)] = downstream.sha(report)
        state["jobs"][identifier] = row
        return identifier

    for protocol in protocols:
        for candidate in candidates:
            if protocol["task"] not in candidate["tasks"]:
                continue
            cell = f"{protocol['id']}/{candidate['id']}"
            trial = {"id": "trial_000", "parameters": {}, "is_default": True}
            state["cells"][cell] = {
                "protocol": copy.deepcopy(protocol),
                "candidate": copy.deepcopy(candidate),
                "trials": [trial],
                "rungs": {"0": {"epochs": 1, "successful": 1, "promoted": ["trial_000"]}},
            }
            selection_jobs = [job(cell, "search", 100), job(cell, "confirmation", 101)]
            selected = {
                "trial": trial,
                "mean_two_seed_validation_score": 1.0,
                "selection_jobs": selection_jobs,
            }
            selection_path = root / "selections" / f"{cell.replace('/', '__')}.json"
            dump(selection_path, selected)
            state["cells"][cell].update(
                selected=selected, selection_sha256=downstream.sha(selection_path)
            )
            for variant in ("default", "tuned"):
                for seed in (1, 2, 3):
                    job(cell, f"final_{variant}", seed)
    sources = {
        str(path): downstream.sha(path)
        for path in (downstream.PROJECT / "src/virtual_imu").rglob("*.py")
    }
    state["contract"] = {
        "config": matrix_config,
        "source_sha256": sources,
        "cells": list(state["cells"]),
        "smoke": False,
    }
    dump(root / "pipeline.json", state)
    return root, downstream_config


def test_full_synthetic_cli_runs_five_views_and_resumes_without_overwrite(tmp_path):
    downstream = module()
    root, config = _complete_synthetic_pipeline(tmp_path)
    out = tmp_path / "downstream_results"
    auxiliary = tmp_path / "auxiliary_postprocessor" / "scripts"
    auxiliary.mkdir(parents=True)
    for filename in ("evaluate_reviewer_downstream.py", "evaluate_downstream_benchmark.py"):
        shutil.copyfile(downstream.PROJECT / "scripts" / filename, auxiliary / filename)
    assert not (auxiliary.parent / "src").exists()
    assert not (auxiliary.parent / "pyproject.toml").exists()
    command = [
        sys.executable,
        str(auxiliary / "evaluate_reviewer_downstream.py"),
        "--pipeline-root",
        str(root),
        "--config",
        str(config),
        "--out",
        str(out),
    ]
    first = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240
    )
    assert first.returncode == 0, first.stdout + first.stderr
    assert "FROZEN_REVIEWER_DOWNSTREAM_CONTRACT_SHA256=" in first.stdout
    completion = json.loads((out / "complete.json").read_text(encoding="utf-8"))
    for protocol in downstream.UNITS:
        metadata = json.loads(
            (out / "units" / protocol / "metrics.json").read_text(encoding="utf-8")
        )
        assert metadata["stream_count"] == 70
        assert metadata["method_variants_total"] == 30
        assert metadata["neural_optimizer_runs"] == 60
        assert metadata["support"]["attitude"]["common_valid_rows"] == 360
        summary = pd.read_csv(out / "units" / protocol / "summary.csv")
        assert (
            summary.loc[summary.task == "attitude", "endpoint_rmse_deg_mean"].dropna().max() < 1e-8
        )
        assert (
            summary.loc[summary.task == "strapdown", "position_endpoint_rmse_m_mean"].dropna().max()
            < 1e-8
        )
    before = {str(path): downstream.sha(path) for path in out.glob("units/*/*") if path.is_file()}
    partial = out / ".partial" / "penn_only__attempt_9999"
    partial.mkdir()
    (partial / "PARTIAL.json").write_text('{"status":"abandoned_fixture_attempt"}')
    second = subprocess.run(
        [*command, "--resume"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert (partial / "PARTIAL.json").exists()
    assert {
        str(path): downstream.sha(path) for path in out.glob("units/*/*") if path.is_file()
    } == before
    assert json.loads((out / "complete.json").read_text(encoding="utf-8")) == completion
    (tmp_path / "synthetic_cli_evidence.json").write_text(
        json.dumps(
            {
                "synthetic_only_not_research_results": True,
                "gate_selected_cells": 93,
                "gate_final_evaluations": 558,
                "unit_count": 5,
                "primary_outer_recordings": 4,
                "streams_per_unit": 70,
                "method_variants_per_unit": 30,
                "neural_runs_per_unit": 60,
                "resume_preserved_completed_hashes": True,
                "partial_attempt_preserved": True,
                "auxiliary_scripts_without_src_or_pyproject": True,
                "completion": completion,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
