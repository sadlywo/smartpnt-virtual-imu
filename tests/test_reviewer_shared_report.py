"""Full public-interface fictional fixtures; no real data/model/TEST decoding."""

# ruff: noqa: RUF001
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from test_reviewer_report import fixture as original_fixture
from test_reviewer_report import report

PHYSICAL_GYRO = (
    "gyro_vector_rmse_radps",
    "gyro_vector_mae_radps",
    "gyro_bias_norm_radps",
    "gyro_error_p95_radps",
)
PHYSICAL_ACCEL = (
    "accel_vector_rmse_mps2",
    "accel_vector_mae_mps2",
    "accel_bias_norm_mps2",
    "accel_error_p95_mps2",
)
AXIS_MEASURES = ("rmse", "mae", "mean_error", "error_std")
FULL_SHARED_METRICS = (
    *PHYSICAL_GYRO,
    *PHYSICAL_ACCEL,
    *(
        f"axis_{measure}_{axis}"
        for measure in AXIS_MEASURES
        for axis in ("gx", "gy", "gz", "ax", "ay", "az")
    ),
    "samples",
    "gyro_vector_rmse_radps_relative_identity",
    "accel_vector_rmse_mps2_relative_identity",
    "grouped_relative_score",
)


def shared_fixture(root: Path):
    package, figures = original_fixture(root)
    manifest_path = package / "metadata/package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aux = "server/aux3/runs/shared"
    base = package / aux
    base.mkdir(parents=True)
    synthetic = package / manifest["roots"]["core_pipeline"] / "fixture_payload.json"
    payload_sha = report.sha256_file(synthetic)
    source = "/fiction/" + synthetic.relative_to(package).as_posix()

    def put(name, value):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        report.json_write(path, value)
        return report.sha256_file(path)

    def csvfile(name, rows):
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return {
            "path": name,
            "sha256": report.sha256_file(path),
            "rows": len(rows),
            "columns": columns,
        }

    profile, scope = report.SHARED_PROFILE, report.SHARED_SCOPE
    pipeline_path = package / manifest["roots"]["core_pipeline"] / "pipeline.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    pipeline_sha = report.sha256_file(pipeline_path)
    protocols = report.PROTOCOLS[:6]
    config_sha = put(
        "synthetic_config.json", {"analysis_profile": profile, "synthetic_fixture": True}
    )
    references = {"split_manifest": {"path": source, "sha256": payload_sha}}
    fit_contract = {
        "pipeline_sha256": pipeline_sha,
        "config_path": "/fiction/" + aux + "/synthetic_config.json",
        "config_sha256": config_sha,
        "source_sha256": {source: payload_sha},
        "manifests": {p: references for p in protocols},
    }
    fit_sha = report._shared_digest(fit_contract)
    put("fit_select/contract.json", fit_contract)
    selections, controls, candidates, scopes = {}, [], [], []
    for p in protocols:
        selection = {
            "protocol": p,
            "contract_sha256": fit_sha,
            "source_sha256": fit_contract["source_sha256"],
            "manifests": references,
        }
        name = f"fit_select/{p}/selection.json"
        selections[p] = {"path": name, "sha256": put(name, selection)}
        for method in report.CLASSICAL:
            taps = (
                9
                if method == "learned_fir_causal"
                else 12
                if method == "arx_linear"
                else 1
                if method == "fir_tuned"
                else 3
                if method == "arx_tuned"
                else None
            )
            count = (
                0
                if method == "identity"
                else 6
                if method == "bias_mean"
                else 36 * (taps + int(method.startswith("arx"))) + 6
                if taps
                else 42
            )
            alpha = (
                1e-6
                if method.endswith("tuned")
                else None
                if method in {"identity", "bias_mean", "affine"}
                else 0.01
            )
            kind = (
                "identity"
                if method == "identity"
                else "bias"
                if method == "bias_mean"
                else "linear"
            )
            numeric = {
                "state_kind": kind,
                "weights": [[0.0] * 6 for _ in range((count - 6) // 6)] if kind == "linear" else [],
                "intercept": [] if kind == "identity" else [0.0] * 6,
                "hyperparameters": {"alpha": alpha, "taps": taps},
                "feature_spec": {
                    "taps": taps,
                    "include_delta": method.startswith("arx"),
                    "edge_padding": "first_sample",
                    "initial_delta": 0,
                    "split_local": True,
                    "stream_key": "sequence_id",
                },
                "residual_add_low": kind == "bias",
                "normalization": {"mode": "none_raw_si", "coefficient_count": 0},
                "arithmetic": "weights_bias",
            }
            key = {
                "protocol": p,
                "information_scope": scope,
                "canonical_method": method,
                "variant": "shared",
            }
            numeric_sha = report._shared_digest(numeric)
            state_path = f"fit_select/{p}/{method}.state.json"
            state_sha = put(
                state_path,
                {
                    "synthetic_fixture": True,
                    "schema_version": 1,
                    **key,
                    **numeric,
                    "input_channels": 6,
                    "output_channels": 6,
                    "numeric_sha256": numeric_sha,
                    "coefficient_count": count,
                    "coefficient_bytes": count * 8,
                    "normalizer_count": 0,
                    "normalizer_bytes": 0,
                },
            )
            key = {
                "protocol": p,
                "information_scope": scope,
                "canonical_method": method,
                "variant": "shared",
            }
            controls.append(
                {
                    **key,
                    "n": 1,
                    "sd": None,
                    "state_path": state_path,
                    "state_sha256": state_sha,
                    "numeric_sha256": numeric_sha,
                    "train_row_sha256": payload_sha,
                    "validation_row_sha256": payload_sha,
                    "identity_alias": method == "identity",
                    "learned_state": method != "identity",
                    "normalization": "none_raw_si",
                    "normalizer_count": 0,
                    "coefficient_count": count,
                    "coefficient_bytes": count * 8,
                    "serialized_state_bytes": (base / state_path).stat().st_size,
                    "taps": taps,
                    "alpha": alpha,
                }
            )
            scopes.append(
                {
                    **key,
                    "n": 1,
                    "sd": None,
                    "identity_routes_state": False,
                    "input_scope": "six_LOW_and_same_stream_LOW_history",
                }
            )
        order = 0
        for method, taps_grid in (
            ("ridge_tuned", [None]),
            ("fir_tuned", [1, 3, 5, 9, 17, 33, 65]),
            ("arx_tuned", [3, 6, 12, 24, 48]),
        ):
            for taps in taps_grid:
                for alpha in [1e-6, 1e-4, 0.01, 0.1, 1, 10, 100]:
                    candidates.append(
                        {
                            "protocol": p,
                            "information_scope": scope,
                            "canonical_method": method,
                            "variant": "shared",
                            "taps": taps,
                            "alpha": alpha,
                            "order": order,
                            "finite": True,
                            "error": None,
                            "validation_score": 1 + order * 0.01,
                        }
                    )
                    order += 1
    freeze_sha = put(
        "fit_select/selection_set.json",
        {
            "schema_version": 1,
            "analysis_profile": profile,
            "status": "frozen_before_any_test_load",
            "contract_sha256": fit_sha,
            "protocols": selections,
            "synthetic_fixture": True,
        },
    )
    descriptors, stages = {}, {}

    def stage(name, contract_sha, tables):
        stage_tables = {
            role: csvfile(name + "/" + role + ".csv", rows) for role, rows in tables.items()
        }
        summary_path = name + "/stage_summary.json"
        summary_sha = put(
            summary_path,
            {
                "schema_version": 1,
                "analysis_profile": profile,
                "status": "complete",
                "stage": name,
                "selection_set_sha256": freeze_sha,
                "contract_sha256": contract_sha,
                "tables": stage_tables,
            },
        )
        files = {
            p.relative_to(base).as_posix(): report.sha256_file(p)
            for p in (base / name).rglob("*")
            if p.is_file()
        }
        manifest_name = name + "/artifact_manifest.json"
        manifest_sha = put(
            manifest_name,
            {
                "schema_version": 1,
                "analysis_profile": profile,
                "status": "complete",
                "root_kind": "aux3_run",
                "contract_sha256": contract_sha,
                "files": files,
            },
        )
        descriptors.update(stage_tables)
        stages[name] = {
            "path": manifest_name,
            "sha256": manifest_sha,
            "summary_path": summary_path,
            "summary_sha256": summary_sha,
        }

    stage(
        "fit_select",
        fit_sha,
        {
            "control_registry": controls,
            "shared_state_inventory": controls,
            "method_scope": scopes,
            "candidate_history": candidates,
        },
    )
    signals, neural, metrics, aliases = [], [], [], []
    put(
        "signal_evaluation/synthetic_predictions.json",
        {"synthetic_fixture": True, "not_a_prediction_payload": True},
    )
    for c in controls:
        key = {k: c[k] for k in report.METHOD_KEY}
        for split in ("validation", "test"):
            signals.append(
                {
                    **key,
                    "split": split,
                    "n": 1,
                    "sd": None,
                    "rows": 800,
                    "row_sha256": payload_sha,
                    "selection_set_sha256": freeze_sha,
                    "prediction_path": "signal_evaluation/synthetic_predictions.json",
                }
            )
            for metric in FULL_SHARED_METRICS:
                metrics.append(
                    {
                        **key,
                        "split": split,
                        "n": 1,
                        "sd": None,
                        "metric_scope": "dataset",
                        "dataset_id": "fixture",
                        "low_sensor": "__all_devices__",
                        "high_sensor": "__all_devices__",
                        "sequence_id": "__all_sequences__",
                        "metric": metric,
                        "value": 800
                        if metric == "samples"
                        else 1.5
                        if c["canonical_method"] == "identity"
                        else 1.7,
                    }
                )
        penn = c["protocol"] not in {"pooled", "alimu_only"}
        aliases.append(
            {
                **key,
                "n": 1,
                "sd": None,
                "alias_status": "exact_alias"
                if penn
                else "common_identity"
                if c["identity_alias"]
                else "not_applicable_multi_pair_bank",
                "shared_state_sha256": c["state_sha256"] if penn else "",
                "bank_state_sha256": payload_sha if penn else "",
                "rtol": 1e-10 if penn else None,
                "atol": 1e-12 if penn else None,
                "proof": json.dumps(
                    dict.fromkeys(
                        (
                            "same_protocol",
                            "train_rows_exact",
                            "validation_rows_exact",
                            "grid_order_exact",
                            "tie_rule_exact",
                            "bank_own_selection_valid",
                        ),
                        True,
                    )
                )
                if penn
                else "",
            }
        )
    for jid, job in pipeline["jobs"].items():
        p = job["cell"].split("/")[0]
        if p not in protocols:
            continue
        for split in ("validation", "test"):
            neural.append(
                {
                    "protocol": p,
                    "job_id": jid,
                    "cell": job["cell"],
                    "phase": job["phase"],
                    "seed": job["seed"],
                    "split": split,
                    "rows": 800,
                    "physical_sha256": payload_sha,
                    "prediction_path": source,
                    "prediction_sha256": payload_sha,
                }
            )
    audit_sha = put("signal_evaluation/original_full_audit.json", {"synthetic_fixture": True})
    stage(
        "signal_evaluation",
        fit_sha,
        {
            "metrics_long": metrics,
            "signal_registry": signals,
            "neural_support_registry": neural,
            "alias_audit": aliases,
        },
    )
    downstream, windows, segments, proofs = [], [], [], []
    for p, recording in report.DOWNSTREAM_VIEWS.items():
        for task in ("attitude", "strapdown"):
            proofs.append(
                {
                    "protocol": p,
                    "task": task,
                    "recording": recording,
                    "identity_alias_verified": True,
                    "independent_gt_sha256": payload_sha,
                }
            )
            for method in report.CLASSICAL:
                key = {
                    "protocol": p,
                    "information_scope": scope,
                    "canonical_method": method,
                    "variant": "shared",
                    "task": task,
                    "n": 1,
                    "sd": None,
                    "recording": recording,
                }
                for samples, horizon in zip(report.HORIZON_SAMPLES, report.HORIZONS, strict=True):
                    value = 1 if method == "identity" else 1.4
                    downstream.append(
                        {
                            **key,
                            "samples": samples,
                            "horizon_s": horizon,
                            "optimizer_runs": 1,
                            "count": 1,
                            "status": "supported",
                            "endpoint_rmse_deg": value if task == "attitude" else None,
                            "position_endpoint_rmse_m": value if task == "strapdown" else None,
                        }
                    )
                    windows.append(
                        {
                            **key,
                            "samples": samples,
                            "sequence_id": recording,
                            "segment": 0,
                            "start_s": 0,
                            "end_s": horizon,
                            "synthetic_fixture": True,
                        }
                    )
                segments.append(
                    {
                        **key,
                        "sequence_id": recording,
                        "segment": 0,
                        "start_s": 0,
                        "end_s": 19.975,
                        "synthetic_fixture": True,
                    }
                )
    put("downstream/support_proofs.json", proofs)
    downstream_contract = {
        "fit_selection_set_sha256": freeze_sha,
        "signal_manifest_sha256": stages["signal_evaluation"]["sha256"],
    }
    put("downstream/contract.json", downstream_contract)
    stage(
        "downstream",
        report._shared_digest(downstream_contract),
        {
            "downstream_summary": downstream,
            "downstream_windows": windows,
            "downstream_segments": segments,
        },
    )
    inventory, totals, timings, workload, raw = [], [], [], [], []
    for c in controls:
        for s, variant in ((scope, "shared"), ("known_device_bank", "bank")):
            key = {k: c[k] for k in report.METHOD_KEY}
            key.update(information_scope=s, variant=variant)
            inventory.append(
                {
                    **key,
                    "normalizer_count": 0,
                    "normalizer_bytes": 0,
                    "coefficient_count": c["coefficient_count"],
                    "coefficient_bytes": c["coefficient_bytes"],
                }
            )
            totals.append(
                {
                    **key,
                    "pair_count": 1,
                    "bank_coefficient_count": c["coefficient_count"],
                    "bank_coefficient_bytes": c["coefficient_bytes"],
                }
            )
            for w in (80, 160, 320):
                for b in (1, 128):
                    ident = "/".join(str(key[k]) for k in report.METHOD_KEY) + f"/{w}/{b}"
                    timing = {
                        **key,
                        "n": 1,
                        "sd": None,
                        "device": "cpu",
                        "dtype": "float64",
                        "window_size": w,
                        "batch_size_per_pair": b,
                        "warmup": 30,
                        "iterations": 100,
                        "repeats": 3,
                        "p50_ms": 0.1,
                        "p95_ms": 0.2,
                        "call_distribution_sd_ms": 0.01,
                        "loop_throughput_windows_per_second": 1000,
                        "timing_id": ident,
                        "pair_count": 1,
                        "windows_per_call": b,
                        "synthetic_fixture": True,
                    }
                    timings.append({**timing, "measurement_kind": "per_state"})
                    workload.append({**timing, "measurement_kind": "full_pair_workload"})
                    raw.append(
                        {"timing_id": ident, "value": 0.1, "unit": "ms", "synthetic_fixture": True}
                    )
    cpu_contract = {
        "selection_set_sha256": freeze_sha,
        "fit_contract_sha256": fit_sha,
        "pipeline_sha256": pipeline_sha,
        "upstream_manifests": {s: stages[s]["sha256"] for s in report.SHARED_STAGES[:3]},
    }
    put("cpu_latency/contract.json", cpu_contract)
    stage(
        "cpu_latency",
        report._shared_digest(cpu_contract),
        {
            "state_inventory": inventory,
            "bank_parameter_totals": totals,
            "latency_summary": timings,
            "latency_raw": raw,
            "workload_latency": workload,
        },
    )
    index_sha = put(
        "shared_controls_artifacts.json",
        {
            "schema_version": 1,
            "analysis_profile": profile,
            "status": "complete",
            "counts": report.SHARED_COUNTS,
            "global_selection_set_sha256": freeze_sha,
            "fit_contract_sha256": fit_sha,
            "stage_manifests": stages,
            "tables": descriptors,
            "original": {
                "core_pipeline_sha256": pipeline_sha,
                "selected_cells": 93,
                "evaluated_final_records": 558,
                "neural_transfer_records": 120,
                "downstream_views": 5,
                "original_full_audit_path": "signal_evaluation/original_full_audit.json",
                "original_full_audit_sha256": audit_sha,
            },
        },
    )
    put(
        "shared_controls_state.json",
        {
            "status": "complete",
            "analysis_profile": profile,
            "counts": report.SHARED_COUNTS,
            "global_selection_set_sha256": freeze_sha,
            "shared_controls_artifacts_sha256": index_sha,
        },
    )
    audit_path = package / "metadata/completion_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update(analysis_profile=profile)
    audit["counts"].update(report.EXPORT_SHARED_COUNTS)
    audit["shared_controls_index"] = {
        "source_path": "/fiction/" + aux + "/shared_controls_artifacts.json",
        "sha256": index_sha,
    }
    audit["auxiliary_checks"]["aux3"] = json.loads(
        (base / "shared_controls_artifacts.json").read_text(encoding="utf-8")
    )
    marker_name = "shared_terminal.log"
    (base / marker_name).write_text("REVIEWER_SHARED_CONTROLS_PASSED\n", encoding="utf-8")
    audit["marker_checks"].append(
        {
            "status": "passed",
            "source_path": "/fiction/" + aux + "/" + marker_name,
            "sha256": report.sha256_file(base / marker_name),
        }
    )
    report.json_write(audit_path, audit)
    manifest["roots"].update(aux3_project="server/aux3", aux3=aux)
    manifest["source_roots"].update(aux3_project="/fiction/server/aux3", aux3="/fiction/" + aux)
    manifest["path_mappings"].append(
        {"source_prefix": "/fiction/server/aux3", "mirror_prefix": "server/aux3"}
    )
    manifest.update(analysis_profile=profile)
    manifest["files"] = {
        p.relative_to(package).as_posix(): report.sha256_file(p)
        for p in package.rglob("*")
        if p.is_file() and p != manifest_path
    }
    manifest["file_metadata"] = {
        name: {
            "size_bytes": (package / name).stat().st_size,
            "source_path": None if name.startswith("metadata/") else "/fiction/" + name,
        }
        for name in manifest["files"]
    }
    report.json_write(manifest_path, manifest)
    figure_path = figures / "figure_index.json"
    fig = json.loads(figure_path.read_text(encoding="utf-8"))
    fig.update(
        analysis_profile=profile,
        shared_controls_index_sha256=index_sha,
        shared_coverage={**report.SHARED_COUNTS, "neural_split_support_references": 720},
        manifest_sha256=report.sha256_file(manifest_path),
        source_manifest_sha256=report.sha256_file(manifest_path),
    )
    extra = json.loads(json.dumps(fig["figures"][0]))
    extra.update(
        id="shared_fixture_scope",
        title="共享经典合成接口验证",
        caption="全为合成值；不代表实验结论。",
        coverage={"group": "shared", "information_scope": scope},
        source_tables=[
            {
                "path": aux + "/" + descriptors["metrics_long"]["path"],
                "sha256": descriptors["metrics_long"]["sha256"],
            }
        ],
    )
    # Separate new-profile synthetic QA identity; legacy QA is not reused.
    qa_path = figures / "shared_fixture_scope_qa.json"
    report.json_write(
        qa_path,
        {
            "status": "passed",
            "visual_review_status": "passed",
            "synthetic_fixture": True,
            "analysis_profile": profile,
            "shared_controls_index_sha256": index_sha,
            "temporal_support_sha256": report.sha256_file(report.TEMPORAL_HELPER),
        },
    )
    extra["qa"].update(path=qa_path.name, sha256=report.sha256_file(qa_path))
    fig["figures"].append(extra)
    report.json_write(figure_path, fig)
    return package, figures


def test_shared_complete_roster_and_scoped_summary(tmp_path):
    package, figures = shared_fixture(tmp_path)
    data = report.audit_inputs(package, figures, _allow_synthetic=True)
    assert data["analysis_profile"] == report.SHARED_PROFILE
    assert len(data["shared"]["tables"]["neural_support_registry"]) == 720
    rows, rankings = report.scoped_signal_tables(data)
    shared_rows = [r for r in rows if r["information_scope"] == report.SHARED_SCOPE]
    assert {r["metric"] for r in shared_rows} == set(FULL_SHARED_METRICS)
    for row in shared_rows:
        metric = row["metric"]
        if metric in PHYSICAL_GYRO or metric.endswith(("_gx", "_gy", "_gz")):
            assert row["unit"] == "rad/s"
        elif metric in PHYSICAL_ACCEL or metric.endswith(("_ax", "_ay", "_az")):
            assert row["unit"] == "m/s^2"
        elif metric == "samples":
            assert row["unit"] == "samples"
        else:
            assert row["unit"] == "dimensionless"
    scoped_csv = tmp_path / "scoped_signal_metrics.csv"
    report.write_csv(scoped_csv, rows)
    saved = report._rows(scoped_csv)
    assert all(r["unit"] != "see metric definition" for r in saved)
    assert {r["unit"] for r in saved if r["metric"] == "axis_mae_gx"} == {"rad/s"}
    assert {r["unit"] for r in saved if r["metric"] == "axis_mean_error_az"} == {"m/s^2"}
    assert {r["unit"] for r in saved if r["metric"] == "samples"} == {"samples"}
    originals = data["tables"]["metrics_seed_summary"]
    for original in originals:
        if original["method"] != "model":
            continue
        original_key = (
            original["protocol_id"],
            original["model_family"],
            original["variant"],
            original["split"],
            original["metric"],
            original["dataset_id"],
        )
        mapped = [
            r
            for r in saved
            if r["information_scope"] == "unified_six_low_neural"
            and (
                r["protocol"],
                r["canonical_method"],
                r["variant"],
                r["split"],
                r["metric"],
                r["dataset_id"],
            )
            == original_key
        ]
        assert len(mapped) == 1 and mapped[0]["unit"] == original["unit"]
    methods = [
        r
        for r in rows
        if r["protocol"] == "pooled"
        and r["canonical_method"] == "ridge_tuned"
        and r["split"] == "test"
        and r["metric"] == "gyro_vector_rmse_radps"
    ]
    assert {r["information_scope"] for r in methods} == {report.SHARED_SCOPE, "known_device_bank"}
    assert {float(r["mean"]) for r in methods} == {1.5, 1.7}
    assert all(
        r["information_scope"] != "known_device_bank"
        for r in rankings
        if r["ranking_condition"] == "main_common_information"
    )
    downstream = report.scoped_downstream_rows(data)
    assert all(r["sample_sd"] == "" for r in downstream if r["n"] == 1)
    assert any(
        r["behavior"] == "degraded" and r["information_scope"] == report.SHARED_SCOPE
        for r in downstream
    )


@pytest.mark.parametrize(
    "metric",
    ["unregistered_metric", "gyro_unknown_radps", "axis_unknown_gx", "unknown_relative_identity"],
)
def test_unknown_shared_metric_unit_fails_closed(metric):
    with pytest.raises(ValueError, match="no declared unit"):
        report.shared_metric_unit(metric)
    row = {
        "protocol": "pooled",
        "information_scope": report.SHARED_SCOPE,
        "canonical_method": "ridge_tuned",
        "variant": "shared",
        "split": "test",
        "metric_scope": "dataset",
        "metric": metric,
        "value": 1.0,
        "n": 1,
        "sd": "",
    }
    with pytest.raises(ValueError, match="no declared unit"):
        report.scoped_signal_tables(
            {"tables": {"metrics_seed_summary": []}, "shared": {"tables": {"metrics_long": [row]}}}
        )


def test_original_package_cannot_impersonate_shared(tmp_path):
    package, figures = original_fixture(tmp_path)
    path = package / "metadata/package_manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["analysis_profile"] = report.SHARED_PROFILE
    report.json_write(path, value)
    with pytest.raises(ValueError, match=r"profile differs|incomplete scientific scope|fourth"):
        report.audit_inputs(package, figures, _allow_synthetic=True)


@pytest.mark.parametrize(
    "target", ["missing_ref", "duplicate_slot", "shared_seed_sd", "old_figures", "alias_proof"]
)
def test_shared_semantic_mutations_fail_closed(tmp_path, target):
    package, figures = shared_fixture(tmp_path)
    data = report.audit_inputs(package, figures, _allow_synthetic=True)
    shared = data["shared"]
    if target == "old_figures":
        p = figures / "figure_index.json"
        value = json.loads(p.read_text(encoding="utf-8"))
        value.pop("shared_coverage")
        report.json_write(p, value)
        with pytest.raises(ValueError, match="own figure coverage"):
            report._figures_audit(data["pkg"], figures, True, shared)
        return
    role = {
        "missing_ref": "neural_support_registry",
        "duplicate_slot": "signal_registry",
        "shared_seed_sd": "control_registry",
        "alias_proof": "alias_audit",
    }[target]
    table = shared["tables"][role]
    if target == "missing_ref":
        table.pop()
    elif target == "duplicate_slot":
        table[-1] = dict(table[0])
    elif target == "shared_seed_sd":
        table[0]["n"], table[0]["sd"] = "3", "0.1"
    else:
        row = next(r for r in table if r["alias_status"] == "exact_alias")
        row["proof"] = json.dumps({"same_protocol": False})
    descriptor = shared["descriptors"][role]
    path = shared["root"] / descriptor["path"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=descriptor["columns"])
        writer.writeheader()
        writer.writerows(table)
    # Re-sign every affected layer: fail on semantic rosters, not merely checksum.
    rebind_shared(package, figures, role)
    with pytest.raises(ValueError, match=r"tuple coverage|n=1|alias provenance"):
        report.audit_inputs(package, figures, _allow_synthetic=True)


def rebind_shared(package, figures, role):
    """Re-sign synthetic metadata so mutation tests reach scientific semantics."""
    mp = package / "metadata/package_manifest.json"
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    root = package / manifest["roots"]["aux3"]
    index_path = root / "shared_controls_artifacts.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    descriptor = index["tables"][role]
    table = report._rows(root / descriptor["path"])
    descriptor.update(sha256=report.sha256_file(root / descriptor["path"]), rows=len(table))
    stage = descriptor["path"].split("/")[0]
    sp = root / stage / "stage_summary.json"
    summary = json.loads(sp.read_text(encoding="utf-8"))
    summary["tables"][role] = descriptor
    report.json_write(sp, summary)
    ap = root / stage / "artifact_manifest.json"
    artifact = json.loads(ap.read_text(encoding="utf-8"))
    artifact["files"] = {
        p.relative_to(root).as_posix(): report.sha256_file(p)
        for p in (root / stage).rglob("*")
        if p.is_file() and p != ap
    }
    report.json_write(ap, artifact)
    index["stage_manifests"][stage].update(
        sha256=report.sha256_file(ap), summary_sha256=report.sha256_file(sp)
    )
    # Mutated fit/signal stage has a downstream DAG; propagate actual manifest SHA.
    for downstream_stage in ("downstream", "cpu_latency"):
        cp = root / downstream_stage / "contract.json"
        contract = json.loads(cp.read_text(encoding="utf-8"))
        if downstream_stage == "downstream":
            contract["signal_manifest_sha256"] = index["stage_manifests"]["signal_evaluation"][
                "sha256"
            ]
        else:
            contract["upstream_manifests"] = {
                s: index["stage_manifests"][s]["sha256"] for s in report.SHARED_STAGES[:3]
            }
        report.json_write(cp, contract)
        sp = root / downstream_stage / "stage_summary.json"
        summary = json.loads(sp.read_text(encoding="utf-8"))
        summary["contract_sha256"] = report._shared_digest(contract)
        report.json_write(sp, summary)
        ap = root / downstream_stage / "artifact_manifest.json"
        artifact = json.loads(ap.read_text(encoding="utf-8"))
        artifact["contract_sha256"] = summary["contract_sha256"]
        artifact["files"] = {
            p.relative_to(root).as_posix(): report.sha256_file(p)
            for p in (root / downstream_stage).rglob("*")
            if p.is_file() and p != ap
        }
        report.json_write(ap, artifact)
        index["stage_manifests"][downstream_stage].update(
            sha256=report.sha256_file(ap), summary_sha256=report.sha256_file(sp)
        )
    report.json_write(index_path, index)
    state_path = root / "shared_controls_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["shared_controls_artifacts_sha256"] = report.sha256_file(index_path)
    report.json_write(state_path, state)
    audit_path = package / "metadata/completion_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["shared_controls_index"]["sha256"] = report.sha256_file(index_path)
    audit["auxiliary_checks"]["aux3"] = index
    report.json_write(audit_path, audit)
    manifest["files"] = {
        p.relative_to(package).as_posix(): report.sha256_file(p)
        for p in package.rglob("*")
        if p.is_file() and p != mp
    }
    for name in manifest["files"]:
        manifest["file_metadata"][name]["size_bytes"] = (package / name).stat().st_size
    report.json_write(mp, manifest)
    fp = figures / "figure_index.json"
    figures_index = json.loads(fp.read_text(encoding="utf-8"))
    figures_index.update(
        manifest_sha256=report.sha256_file(mp),
        source_manifest_sha256=report.sha256_file(mp),
        shared_controls_index_sha256=report.sha256_file(index_path),
    )
    for f in figures_index["figures"]:
        for source in f["source_tables"]:
            source["sha256"] = report.sha256_file(package / source["path"])
    report.json_write(fp, figures_index)


def test_shared_pdf_and_resume(tmp_path):
    package, figures = shared_fixture(tmp_path)
    out = tmp_path / "report"
    value = report.build_report(package, figures, out, _allow_synthetic=True)
    assert (
        value["shared_logical_controls"] == 54 and value["neural_split_support_references"] == 720
    )
    text = (out / "report.md").read_text(encoding="utf-8")
    for phrase in ("known-device bank", "N=floor", "not_alias", "P×B", "共享经典", "0.1s"):
        assert phrase in text if phrase != "N=floor" else "C=floor(N/40)*40" in text
    assert (out / "tables/scoped_signal_rankings.csv").is_file()
    helper_sha = report.sha256_file(report.TEMPORAL_HELPER)
    assert value["temporal_support_sha256"] == helper_sha
    assert report.sha256_file(out / "processing/reviewer_temporal_support.py") == helper_sha
    costs = report._rows(out / "tables/neural_latency_contexts.csv")
    assert len(costs) == 1674 and {r["canonical_method"] for r in costs} == set(report.FAMILIES) | {
        "tartanimu_window_adapter"
    }
    assert all(r["temporal_class"] == "window_endpoint" for r in costs if r["task"] == "navigation")
    assert all(r["timing_representative_entry_id"] for r in costs)
    assert {r["batch_size"] for r in costs if r["device"] == "cpu"} == {"1"}
    assert {r["batch_size"] for r in costs if r["device"] == "cuda:0"} == {"1", "128"}
    assert "未测（冻结CPU协议）" in text
    index = report._rows(out / "tables/table_index.csv")
    assert all(r["temporal_support_sha256"] == helper_sha for r in index)
    assert any(
        r["table"] == "neural_parameter_contexts" and r["temporal_annotations"] == "True"
        for r in index
    )
    assert report.build_report(package, figures, out, _allow_synthetic=True, resume=True) == value
