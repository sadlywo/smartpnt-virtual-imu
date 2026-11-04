"""Full business-scope fictional fixtures; never train or read real test data."""

# ruff: noqa: RUF001
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "reviewer_report", SCRIPTS / "build_reviewer_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture(root: Path):
    """Build 93/558/120/5 explicitly synthetic identities, not fake real scores."""
    package, figures = root / "mirror", root / "figures"
    package.mkdir(parents=True)
    figures.mkdir()
    roots = {
        "core_project": "server/core_project",
        "core_pipeline": "server/core_project/runs/hpo",
        "aux1_project": "server/aux1",
        "aux1": "server/aux1/runs/post",
        "aux2_project": "server/aux2",
        "aux2": "server/aux2/runs/analysis",
    }
    sources = {key: "/fiction/" + value for key, value in roots.items()}

    def put(name, value):
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return path

    def csvfile(name, rows):
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    core, a1, a2 = roots["core_pipeline"], roots["aux1"], roots["aux2"]
    payload = put(core + "/fixture_payload.json", {"synthetic": True, "not_a_model": True})
    digest = report.sha256_file(payload)
    source_payload = "/fiction/" + core + "/fixture_payload.json"
    cells, jobs, registry, inventory, checkpoints, predictions, fits, selections = (
        {},
        {},
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for cell in sorted(report.EXPECTED_CELLS):
        protocol, family = cell.split("/")
        params = {
            "training.lr": 0.001,
            "training.weight_decay": 0.00001,
            "loss_profile": "huber_only" if protocol.startswith("navigation") else "composite",
        }
        cells[cell] = {
            "trials": [{"id": "trial_000", "parameters": params}],
            "selected": {"trial": {"id": "trial_001", "parameters": params}},
            "selection_sha256": digest,
        }
        selections.append(
            {"cell": cell, "status": "passed", "path": source_payload, "sha256": digest}
        )
        for variant in report.VARIANTS:
            for seed in report.SEEDS:
                jid = cell.replace("/", "__") + f"__{variant}__{seed}"
                task = "navigation" if protocol.startswith("navigation") else "paired"
                train = {
                    "epochs": 80,
                    "loss": "huber",
                    "gyro_signal_weight": 2.0,
                    "accel_signal_weight": 1.0,
                    "derivative_weight": 0.02,
                    "spectral_weight": 0.02,
                    "preintegration_weight": 0.005,
                    "allan_weight": 0.0,
                    "lr": 0.001,
                    "weight_decay": 0.00001,
                    "lr_scheduler": "cosine_warmup",
                    "warmup_epochs": 5,
                    "min_lr": 0.000001,
                    "grad_clip_norm": 1.0,
                    "early_stopping_patience": 15,
                    "checkpoint_metric": "val_vector_rmse"
                    if task == "navigation"
                    else "val_grouped_score",
                }
                config_member = core + "/configs/" + jid + ".yaml"
                config_path = put(
                    config_member,
                    {
                        "model": {
                            "type": {
                                "deepgyro_adapter": "deepgyro_backbone",
                                "tlio_resnet18_adapter": "tlio_resnet18",
                                "tartanimu_window_adapter": "tartanimu_window",
                            }.get(family, family),
                            "in_channels": 6,
                            "out_channels": 3 if task == "navigation" else 6,
                            "target_mode": "window" if task == "navigation" else "residual",
                        },
                        "training": train,
                    },
                )
                config_sha = report.sha256_file(config_path)
                config_source = "/fiction/" + config_member
                job = {
                    "id": jid,
                    "cell": cell,
                    "phase": "final_" + variant,
                    "trial_id": "trial_000" if variant == "default" else "trial_001",
                    "seed": seed,
                    "status": "complete",
                    "evaluation_status": "complete",
                    "checkpoint_sha256": digest,
                    "config_sha256": config_sha,
                    "run_config_sha256": config_sha,
                    "config": config_source,
                    "run_dir": "/fiction/" + core,
                    "evaluation_sha256": digest,
                    "evaluation_artifact_sha256": {source_payload: digest},
                }
                jobs[jid] = job
                registry.append(
                    {
                        "job_id": jid,
                        "task": task,
                        "protocol_id": protocol,
                        "training_scope": protocol,
                        "model_family": family,
                        "variant": variant,
                        "seed": seed,
                        "best_epoch": 2,
                        "actual_epochs": 4,
                        "epoch_budget": 80,
                        "early_stopped": True,
                        "validation_criterion": train["checkpoint_metric"],
                        "validation_best_metric": 1.0,
                        "elapsed_seconds": 2.0,
                        "window_points": 80,
                        "run_dir": job["run_dir"],
                        "config": config_source,
                        "config_sha256": config_sha,
                        "checkpoint_sha256": digest,
                    }
                )
                inventory.append(
                    {
                        "entry_id": jid,
                        "architecture_id": f"{task}__{family}__w80",
                        "task": task,
                        "protocol_id": protocol,
                        "model_family": family,
                        "variant": variant,
                        "checkpoint_sha256": digest,
                        "seed": seed,
                        "parameters": 123,
                        "trainable_parameters": 123,
                        "input_channels": 6,
                        "output_channels": 3 if task == "navigation" else 6,
                        "window_size": 80,
                    }
                )
                checkpoints.append(
                    {
                        "job_id": jid,
                        "phase": job["phase"],
                        "finite": True,
                        "source_path": source_payload,
                        "sha256": digest,
                        "tensor_count": 1,
                        "tensor_elements": 123,
                    }
                )
                for split in ("validation", "test"):
                    predictions.append(
                        {
                            "job_id": jid,
                            "split": split,
                            "finite": True,
                            "identity_nonnull": True,
                            "rows": 800,
                            "columns": 20,
                            "source_path": source_payload,
                            "sha256": digest,
                        }
                    )
                fits.append(
                    {
                        "job_id": jid,
                        "phase": job["phase"],
                        "status": "passed",
                        "config_path": config_source,
                        "config_sha256": config_sha,
                        "validation": {"finite_and_selected_minimum": True},
                        "evaluation": {
                            "status": "passed",
                            "metrics": {"numeric_finite": True},
                            "artifact_sha256": {source_payload: digest},
                        },
                    }
                )
    config = {
        "seed": 20261024,
        "confirmation_seed": 20261025,
        "seeds": report.SEEDS,
        "search": {
            "trials": 24,
            "rung_epochs": [8, 24, 80],
            "rung_keep": [24, 8, 2],
            "confirmation_candidates": 2,
        },
    }
    pipeline = {
        "status": "complete",
        "smoke": False,
        "git_commit": "synthetic-not-a-real-commit",
        "selection_uses_test": False,
        "contract": {"config": config, "source_sha256": {source_payload: digest}},
        "cells": cells,
        "jobs": jobs,
        "final_seeds": report.SEEDS,
        "final_variants": report.VARIANTS,
    }
    pipeline_path = put(core + "/pipeline.json", pipeline)
    pipeline_sha = report.sha256_file(pipeline_path)
    summaries = []
    for protocol in report.PROTOCOLS:
        for family in report.FAMILIES + (
            ["tartanimu_window_adapter"] if protocol.startswith("navigation") else []
        ):
            for variant in report.VARIANTS:
                metric, unit = (
                    ("vector_rmse_m", "m")
                    if protocol.startswith("navigation")
                    else ("gyro_vector_rmse_radps", "rad/s")
                )
                summaries.append(
                    {
                        "task": "navigation" if protocol.startswith("navigation") else "paired",
                        "protocol_id": protocol,
                        "training_scope": protocol,
                        "split": "test",
                        "aggregation_level": "dataset",
                        "dataset_id": "fixture",
                        "low_sensor": "__all_devices__",
                        "high_sensor": "__all_devices__",
                        "sequence_id": "__all_sequences__",
                        "original_sequence_id": "",
                        "physical_group_id": "",
                        "method": "model",
                        "model_family": family,
                        "variant": variant,
                        "metric": metric,
                        "unit": unit,
                        "mean": 1.0 if variant == "default" else 1.1,
                        "sample_sd": 0.1,
                        "n": 3,
                        "optimization_seed_count": 3,
                        "support_count": 800,
                        "physical_recording_count": 1,
                    }
                )
                if not protocol.startswith("navigation"):
                    summaries.append(
                        {**summaries[-1], "metric": "accel_vector_rmse_mps2", "unit": "m/s^2"}
                    )
    dedup = []
    for protocol in report.PROTOCOLS:
        task = "navigation" if protocol.startswith("navigation") else "paired"
        sources_for_protocol = [r["job_id"] for r in registry if r["protocol_id"] == protocol]
        methods = (
            ["zero", "train_mean", "ridge", "ridge_tuned"]
            if task == "navigation"
            else report.CLASSICAL
        )
        for split in ("validation", "test"):
            for method in methods:
                dedup.append(
                    {
                        "task": task,
                        "protocol_id": protocol,
                        "split": split,
                        "method": method,
                        "physical_rows_sha256": digest,
                        "predictions_sha256": digest,
                        "saved_metrics_sha256": digest,
                        "representative_job_id": sources_for_protocol[0],
                        "source_job_ids": sources_for_protocol,
                    }
                )
                for metric in (
                    ["vector_rmse_m"]
                    if task == "navigation"
                    else ["gyro_vector_rmse_radps", "accel_vector_rmse_mps2"]
                ):
                    summaries.append(
                        {
                            **summaries[0],
                            "task": task,
                            "protocol_id": protocol,
                            "training_scope": protocol,
                            "split": split,
                            "method": method,
                            "model_family": method,
                            "variant": "classical",
                            "metric": metric,
                            "unit": report.PRIMARY[metric],
                            "mean": 1.5,
                            "sample_sd": "",
                            "n": 1,
                            "optimization_seed_count": 0,
                        }
                    )
    table_root = a2 + "/accuracy_tables/"
    csvfile(table_root + "run_registry.csv", registry)
    csvfile(table_root + "metrics_seed_summary.csv", summaries)
    long_rows = []
    for row in summaries:
        for seed, offset in (
            zip(report.SEEDS, (-0.1, 0, 0.1), strict=True)
            if row["method"] == "model"
            else [(None, 0)]
        ):
            long_rows.append(
                {
                    **{k: row[k] for k in report.METRIC_GROUPS},
                    "seed": seed,
                    "value": row["mean"] + offset,
                }
            )
    csvfile(table_root + "metrics_long.csv", long_rows)
    csvfile(
        table_root + "all_fit_jobs.csv",
        [
            {"id": j["id"], "status": "complete", "phase": j["phase"], "elapsed_seconds": 2.0}
            for j in jobs.values()
        ],
    )
    for name in (
        "matched_seed_deltas",
        "matched_seed_delta_summary",
        "relative_identity_percent",
        "relative_identity_seed_summary",
        "model_rankings",
        "rank_correlations",
    ):
        csvfile(table_root + name + ".csv", [{"scope": "synthetic", "value": 1.0}])
    put(
        table_root + "classical_deduplication.json",
        dedup,
    )
    put(
        table_root + "summary.json",
        {
            "status": "complete",
            "runs": 558,
            "fit_jobs": 558,
            "failed_fit_jobs": 0,
            "no_inference_or_fitting": True,
            "source_pipeline_sha256": pipeline_sha,
        },
    )
    transfers = [
        {
            "direction": direction,
            "model_family": family,
            "variant": variant,
            "seed": seed,
            "selection_uses_target_labels": False,
            "checkpoint": source_payload,
            "checkpoint_sha256": digest,
        }
        for direction in ("alimu_to_penn", "penn_to_alimu")
        for family in report.FAMILIES
        for variant in report.VARIANTS
        for seed in report.SEEDS
    ]
    put(
        a1 + "/transfer/transfer_metrics.json",
        {"status": "complete", "records": 120, "source_model_records": transfers},
    )
    csvfile(
        a1 + "/transfer/transfer_seed_summary.csv",
        [
            {
                "direction": t["direction"],
                "model_family": t["model_family"],
                "variant": t["variant"],
                "seeds": 3,
                "gyro_vector_rmse_radps_mean": 1.1,
            }
            for t in transfers[::3]
        ],
    )
    downstream = []
    for protocol, recording in report.DOWNSTREAM_VIEWS.items():
        mapping = {}
        for family in report.FAMILIES:
            for variant in report.VARIANTS:
                for seed in report.SEEDS:
                    prefix = f"{family}_{variant}_{seed}"
                    mapping[prefix] = {
                        "method": family + "/" + variant,
                        "family": family,
                        "variant": variant,
                        "seed": seed,
                        "category": "neural",
                    }
        for method in [*report.CLASSICAL, "HIGH"]:
            mapping[method] = {
                "method": method,
                "seed": None,
                "category": "reference" if method == "HIGH" else "classical",
            }
        meta = {
            "protocol": protocol,
            "recording": recording,
            "view": "separate_same_recording_pooled_reference"
            if protocol == "pooled"
            else "outer_recording_holdout",
            "methods": mapping,
            "neural_optimizer_runs": 60,
            "neural_method_variants": 20,
            "unique_classical_methods": 9,
            "method_variants_total": 30,
            "stream_count": 70,
            "physical_support": {
                "rows": 800,
                "physical_sha256": digest,
                "source_clock_origin_s": 0.005,
            },
            "independent_gt_path": source_payload,
            "independent_gt_sha256": digest,
            "support": {
                "attitude": {"common_valid_rows": 800},
                "strapdown": {"common_valid_rows": 800},
            },
            "propagation_protocol": {"attitude": {}, "strapdown": {}},
            "model_records": [],
        }
        base = a1 + "/downstream/units/" + protocol + "/"
        put(base + "metrics.json", meta)
        for task in ("attitude", "strapdown"):
            for samples, horizon in zip(report.HORIZON_SAMPLES, report.HORIZONS, strict=True):
                for method in sorted({r["method"] for r in mapping.values()}):
                    row = {
                        "protocol": protocol,
                        "recording": recording,
                        "task": task,
                        "method": method,
                        "samples": samples,
                        "horizon_s": horizon,
                        "optimizer_runs": 3 if "/" in method else 1,
                        "windows_per_run": 1,
                        "endpoint_rmse_deg_mean": 1.0 if method == "identity" else 1.1,
                        "endpoint_rmse_deg_sd": 0.1 if "/" in method else 0,
                        "position_endpoint_rmse_m_mean": 1.0 if method == "identity" else 1.1,
                        "position_endpoint_rmse_m_sd": 0.1 if "/" in method else 0,
                        "horizontal_endpoint_rmse_m_mean": 0.8,
                    }
                    downstream.append(row)
        csvfile(base + "summary.csv", [r for r in downstream if r["protocol"] == protocol])
        for name in ("trials", "window_details", "full_segments"):
            csvfile(
                base + name + ".csv",
                [
                    {
                        "task": "attitude",
                        "method": "identity",
                        "synthetic": True,
                        "endpoint_error_deg": 1.0,
                    }
                ],
            )
    csvfile(a1 + "/downstream/overview/all_fold_summary.csv", downstream)
    put(a1 + "/downstream/complete.json", {"status": "complete"})
    for device in ("cpu", "gpu"):
        base = a1 + "/latency_" + device + "/"
        representatives = {
            arch: min(
                (
                    r
                    for r in inventory
                    if r["architecture_id"] == arch and r["seed"] == report.SEEDS[0]
                ),
                key=lambda r: r["entry_id"],
            )
            for arch in {r["architecture_id"] for r in inventory}
        }
        mapped_inventory = [
            {
                **r,
                "timing_representative_entry_id": representatives[r["architecture_id"]]["entry_id"],
            }
            for r in inventory
        ]
        csvfile(base + "checkpoint_inventory.csv", mapped_inventory)
        records = [
            {
                "architecture_id": arch,
                "representative_entry_id": representative["entry_id"],
                "seed": 20261010,
                "batch_size": b,
                "device": "cpu" if device == "cpu" else "cuda:0",
                "precision": "float32",
                "parameters": 123,
                "wall": {"p50_ms": 0.1, "p95_ms": 0.2, "count": 300},
                "loop_throughput_windows_per_second": 1000.0,
            }
            for arch, representative in sorted(representatives.items())
            for b in ((1,) if device == "cpu" else (1, 128))
        ]
        put(
            base + "latency.json",
            {
                "status": "complete",
                "entries": 558,
                "unique_architectures": len(representatives),
                "source_pipeline_sha256": pipeline_sha,
                "records": records,
                "timing_protocol": {
                    "batch_sizes": [1] if device == "cpu" else [1, 128],
                    "warmup": 30,
                    "iterations": 100,
                    "repeats": 3,
                    "device": "cpu" if device == "cpu" else "cuda:0",
                    "cpu_threads": 1 if device == "cpu" else 4,
                    "input_seed": 20261024,
                    "end_to_end": False,
                },
            },
        )
        csvfile(
            base + "architecture_mapping.csv",
            [
                {
                    key: r[key]
                    for key in (
                        "entry_id",
                        "protocol_id",
                        "model_family",
                        "variant",
                        "seed",
                        "architecture_id",
                        "timing_representative_entry_id",
                    )
                }
                for r in mapped_inventory
            ],
        )
        for name in ("latency_architectures", "latency_raw_timings"):
            csvfile(
                base + name + ".csv",
                [{"architecture_id": "fixture_arch", "synthetic": True, "value": 0.1}],
            )
    classical_root = a2 + "/classical_latency/"
    put(
        classical_root + "latency.json",
        {
            "status": "complete",
            "final_run_mappings": 558,
            "records": [{"method": "identity", "wall": {"p50_ms": 0.1}}],
        },
    )
    csvfile(
        classical_root + "final_run_mapping.csv", [{"entry_id": row["job_id"]} for row in registry]
    )
    for name in ("classical_latency", "classical_latency_raw"):
        csvfile(
            classical_root + name + ".csv",
            [
                {
                    "method": "identity",
                    "device": "cpu",
                    "precision": "NumPy float64",
                    "wall_p50_ms": 0.1,
                    "wall_p95_ms": 0.2,
                }
            ],
        )
    markers = []
    for i, text in enumerate(
        (
            "REVIEWER_CORE_PASSED\nWORKFLOW_EXIT=0\n",
            "REVIEWER_AUXILIARY_PASSED\n",
            "REVIEWER_FINAL_ANALYSIS_PASSED\n",
        )
    ):
        p = package / (core + f"/marker{i}.log")
        p.write_text(text, encoding="utf-8")
        markers.append(
            {
                "status": "passed",
                "source_path": "/fiction/" + p.relative_to(package).as_posix(),
                "sha256": report.sha256_file(p),
            }
        )
    for role, filename, stages in (
        (
            "aux1",
            "postprocessing_state.json",
            ("transfer", "downstream", "latency_cpu", "latency_gpu"),
        ),
        ("aux2", "final_analysis_state.json", ("accuracy_tables", "classical_latency")),
    ):
        base = roots[role]
        stage_rows = {}
        for stage in stages:
            directory = package / base / stage
            stage_rows[stage] = {
                "status": "complete",
                "artifact_sha256": {
                    p.relative_to(directory).as_posix(): report.sha256_file(p)
                    for p in directory.rglob("*")
                    if p.is_file()
                },
            }
        put(
            base + "/" + filename,
            {"status": "complete", "core_pipeline_sha256": pipeline_sha, "stages": stage_rows},
        )
    put(
        "metadata/completion_audit.json",
        {
            "schema_version": 1,
            "status": "passed",
            "synthetic": True,
            "counts": {
                "selected_cells": 93,
                "final_evaluated": 558,
                "transfer_models": 120,
                "downstream_views": 5,
            },
            "source_pipeline": "/fiction/" + core + "/pipeline.json",
            "source_pipeline_sha256": pipeline_sha,
            "source_roster_check": {"status": "passed"},
            "checkpoint_checks": checkpoints,
            "prediction_checks": predictions,
            "fit_checks": fits,
            "selection_checks": selections,
            "marker_checks": markers,
            "auxiliary_checks": {
                role: {name: {"status": "passed"} for name in names}
                for role, names in (
                    ("aux1", ("transfer", "downstream", "latency_cpu", "latency_gpu")),
                    ("aux2", ("accuracy_tables", "classical_latency")),
                )
            },
        },
    )
    files = {
        p.relative_to(package).as_posix(): report.sha256_file(p)
        for p in package.rglob("*")
        if p.is_file()
    }
    put(
        "metadata/package_manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "synthetic": True,
            "roots": roots,
            "source_roots": sources,
            "path_mappings": [
                {"source_prefix": "/fiction/" + roots[key], "mirror_prefix": roots[key]}
                for key in ("core_project", "aux1_project", "aux2_project")
            ],
            "files": files,
            "file_metadata": {
                name: {
                    "size_bytes": (package / name).stat().st_size,
                    "source_path": None if name.startswith("metadata/") else "/fiction/" + name,
                }
                for name in files
            },
        },
    )
    figure_rows = []
    from PIL import Image, ImageDraw

    for kind, ids in (("summary", ["overview"]), ("training", sorted(cells))):
        for i, ident in enumerate(ids):
            name = ("training_" if kind == "training" else "") + str(i)
            p = figures / (name + ".png")
            image = Image.new("RGB", (1000, 500), "white")
            ImageDraw.Draw(image).text(
                (30, 30), "SYNTHETIC FIXTURE ONLY - no real experiment results", fill="black"
            )
            image.save(p)
            formats = {"png": p.name}
            for suffix in ("pdf", "svg", "tiff"):
                target = figures / (name + "." + suffix)
                target.write_text(
                    "synthetic placeholder - report embeds PNG only", encoding="utf-8"
                )
                formats[suffix] = target.name
            qa = {
                "status": "passed",
                "visual_review_status": "passed",
                "synthetic_fixture": True,
                "temporal_support_sha256": report.sha256_file(report.TEMPORAL_HELPER),
            }
            qpath = figures / (name + "_qa.json")
            qpath.write_text(json.dumps(qa), encoding="utf-8")
            coverage = {
                "group": kind,
                "cell": ident,
                "final_run_count": 6,
                "run_ids": [
                    r["job_id"]
                    for r in registry
                    if r["protocol_id"] + "/" + r["model_family"] == ident
                ],
            }
            figure_rows.append(
                {
                    "id": name,
                    "kind": kind,
                    "title": "合成流程验证",
                    "purpose": "layout fixture",
                    "caption": "仅验证排版，不代表真实成绩。",
                    "files": formats,
                    "sha256": {k: report.sha256_file(figures / v) for k, v in formats.items()},
                    "source_tables": [],
                    "source_data": [],
                    "coverage": coverage,
                    "qa": {
                        "status": "passed",
                        "visual_review_status": "passed",
                        "path": qpath.name,
                        "sha256": report.sha256_file(qpath),
                    },
                }
            )
    snapshot = figures / "source_snapshot_reviewer_temporal_support.py"
    snapshot.write_bytes(report.TEMPORAL_HELPER.read_bytes())
    (figures / "figure_index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "synthetic": True,
                "synthetic_fixture": True,
                "temporal_support_sha256": report.sha256_file(report.TEMPORAL_HELPER),
                "temporal_support_source": {
                    "path": snapshot.name,
                    "sha256": report.sha256_file(snapshot),
                },
                "manifest_sha256": report.sha256_file(package / "metadata/package_manifest.json"),
                "source_manifest_sha256": report.sha256_file(
                    package / "metadata/package_manifest.json"
                ),
                "coverage": {"training_cells": 93, "training_final_runs": 558},
                "source_files": {},
                "figures": figure_rows,
                "qa_summary": {
                    "status": "passed",
                    "visual_review_status": "passed",
                    "pdf_render_checked": True,
                    "synthetic_fixture": True,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return package, figures


def test_full_fixture_gates_and_no_formal_synthetic(tmp_path):
    package, figures = fixture(tmp_path)
    with pytest.raises(ValueError, match="Synthetic"):
        report.audit_inputs(package, figures)
    audited = report.audit_inputs(package, figures, _allow_synthetic=True)
    assert len(audited["registry"]) == 558
    assert len(audited["downstream"]) == 1200


def test_gate_missing_view_and_source_mutation(tmp_path):
    package, figures = fixture(tmp_path)
    index = figures / "figure_index.json"
    value = json.loads(index.read_text(encoding="utf-8"))
    value["qa_summary"]["visual_review_status"] = "pending"
    index.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="visual"):
        report.audit_inputs(package, figures, _allow_synthetic=True)
    (package / "server/core_project/runs/hpo/pipeline.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match=r"size changed|checksum changed"):
        report.audit_inputs(package, figures, _allow_synthetic=True)


def test_seed_sd_and_units_no_pooled_duplicate():
    rows = [
        {
            "protocol": p,
            "task": "attitude",
            "method": "gru/tuned",
            "samples": 80,
            "horizon_s": 1.975,
            "endpoint_rmse_deg_mean": 2,
            "endpoint_rmse_deg_sd": 0.2,
            "optimizer_runs": 3,
        }
        for p in report.DOWNSTREAM_VIEWS
    ]
    rows += [
        {**r, "method": "identity", "endpoint_rmse_deg_mean": 1, "optimizer_runs": 1} for r in rows
    ]
    result = report.downstream_comparison(rows)
    assert len([r for r in result if r["view"] == "outer_recording_holdout"]) == 4
    assert len([r for r in result if r["view"] == "same_recording_pooled_reference"]) == 1
    assert all(r["improvement_over_low_pct"] == -100 for r in result)
    with pytest.raises(ValueError, match="nonfinite"):
        report.finite("nan", "fixture")


def test_saved_summary_recompute_uses_sample_sd_and_units():
    base = dict.fromkeys(report.METRIC_GROUPS, "")
    base.update(method="model", metric="gyro_vector_rmse_radps", unit="rad/s")
    long_rows = [
        {**base, "seed": seed, "value": value}
        for seed, value in zip(report.SEEDS, (1, 2, 3), strict=True)
    ]
    summary = {**base, "mean": 2, "sample_sd": 1, "n": 3}
    report.verify_seed_summaries(long_rows, [summary])
    with pytest.raises(ValueError, match="ddof=1"):
        report.verify_seed_summaries(long_rows, [{**summary, "sample_sd": (2 / 3) ** 0.5}])
    with pytest.raises(ValueError, match="physical metric group"):
        report.verify_seed_summaries(long_rows, [{**summary, "unit": "m"}])


def test_fixed_yaml_scalars_are_finite_and_bound_to_trial(tmp_path):
    from types import SimpleNamespace

    config = {
        "model": {"in_channels": 6, "out_channels": 6, "target_mode": "residual"},
        "training": {
            "epochs": 80,
            "lr": 0.001,
            "weight_decay": 0.00001,
            "loss": "huber",
            "gyro_signal_weight": 2.0,
            "accel_signal_weight": 1.0,
            "derivative_weight": 0.02,
            "spectral_weight": 0.02,
            "preintegration_weight": 0.005,
            "allan_weight": 0.0,
            "lr_scheduler": "cosine_warmup",
            "warmup_epochs": 5,
            "min_lr": 0.000001,
            "grad_clip_norm": 1.0,
            "early_stopping_patience": 15,
            "checkpoint_metric": "val_grouped_score",
        },
    }
    path = tmp_path / "frozen.yaml"
    text = "".join(
        section + ":\n" + "".join("  " + k + ": " + json.dumps(v) + "\n" for k, v in values.items())
        for section, values in config.items()
    )
    path.write_text(text, encoding="utf-8")
    pkg = SimpleNamespace(
        resolve=lambda _: path,
        verify_file=lambda p, expected: report.require(
            report.sha256_file(p) == expected, "hash changed"
        ),
        relative_path=lambda _: "frozen.yaml",
    )
    parameters = {
        "training.lr": 0.001,
        "training.weight_decay": 0.00001,
        "loss_profile": "composite",
    }
    cells = {"pooled/gru": {"trials": [{"id": "trial_000", "parameters": parameters}]}}
    job = {
        "config": str(path),
        "config_sha256": report.sha256_file(path),
        "cell": "pooled/gru",
        "phase": "final_default",
        "trial_id": "trial_000",
        "seed": report.SEEDS[0],
    }
    assert report.frozen_training_settings(pkg, {"j": job}, cells)[0]["loss_profile"] == "composite"
    cells["pooled/gru"]["trials"][0]["parameters"]["training.lr"] = 0.002
    with pytest.raises(ValueError, match="differs from selected/default trial"):
        report.frozen_training_settings(pkg, {"j": job}, cells)


def test_required_scope_cannot_be_relaxed_by_fixture_api(tmp_path):
    package, figures = fixture(tmp_path)
    manifest_path = package / "metadata/package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    member = "metadata/completion_audit.json"
    audit_path = package / member
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["counts"]["final_evaluated"] = 552
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    manifest["files"][member] = report.sha256_file(audit_path)
    manifest["file_metadata"][member]["size_bytes"] = audit_path.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete scientific scope"):
        report.audit_inputs(package, figures, _allow_synthetic=True)


def test_pdf_multipage_and_resume_integrity(tmp_path):
    package, figures = fixture(tmp_path)
    out = tmp_path / "report"
    result = report.build_report(package, figures, out, _allow_synthetic=True)
    from pypdf import PdfReader

    reader = PdfReader(out / "report.pdf")
    text = "".join(page.extract_text() for page in reader.pages)
    assert len(reader.pages) >= 5
    assert "合成" in text and "初速度" in text and "退化" in text
    assert any(page.get("/Annots") for page in reader.pages)
    assert result["synthetic_fixture"] is True
    assert report.build_report(package, figures, out, resume=True, _allow_synthetic=True) == result
    (out / "report.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="output checksum"):
        report.build_report(package, figures, out, resume=True, _allow_synthetic=True)
