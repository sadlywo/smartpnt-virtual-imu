"""Synthetic-only validation of final figure coverage, selection and provenance."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_reviewer_figures.py"


def module():
    spec = importlib.util.spec_from_file_location("reviewer_figures", SCRIPT)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def fixture():
    rng = np.random.default_rng(20261024)
    metrics, training, downstream = [], [], []
    methods = [
        ("transformer", "default"),
        ("transformer", "tuned"),
        ("informer", "default"),
        ("informer", "tuned"),
    ]
    methods += [
        (name, "classical")
        for name in [
            "identity",
            "bias_mean",
            "affine",
            "ridge_affine",
            "learned_fir_causal",
            "arx_linear",
            "ridge_tuned",
            "fir_tuned",
            "arx_tuned",
        ]
    ]
    for method, variant in methods:
        for index, metric in enumerate(
            [
                "gyro_vector_rmse_radps",
                "gyro_vector_mae_radps",
                "gyro_bias_norm_radps",
                "accel_vector_rmse_mps2",
                "accel_vector_mae_mps2",
                "accel_bias_norm_mps2",
            ]
        ):
            metrics.append(
                {
                    "task": "paired",
                    "protocol_id": "penn_only",
                    "split": "test",
                    "aggregation_level": "pair",
                    "dataset_id": "penncosyvio",
                    "low_sensor": "LOW",
                    "high_sensor": "HIGH",
                    "method": "model" if variant != "classical" else method,
                    "model_family": method,
                    "variant": variant,
                    "metric": metric,
                    "mean": 0.1 + (index + 1) * 0.02 + (0.05 if variant == "tuned" else 0),
                    "sample_sd": 0.01 if variant != "classical" else np.nan,
                    "n": 3 if variant != "classical" else 1,
                    "unit": "rad/s" if metric.startswith("gyro") else "m/s^2",
                }
            )
    for family in ("transformer", "informer"):
        for variant in ("default", "tuned"):
            for seed in (20261010, 20261011, 20261012):
                for epoch in (1, 2, 3):
                    training.append(
                        {
                            "cell": "penn_only/" + family,
                            "job_id": f"{family}_{variant}_{seed}",
                            "protocol_id": "penn_only",
                            "model_family": family,
                            "task": "paired",
                            "variant": variant,
                            "seed": seed,
                            "epoch": epoch,
                            "train_loss": 2 / epoch,
                            "val_loss": 2.2 / epoch,
                            "criterion": 1.5 / epoch + (0.1 if variant == "tuned" else 0),
                            "criterion_name": "val_score",
                            "loss_recipe": json.dumps(
                                {"mse_weight": 1 if variant == "default" else 0.25}
                            ),
                            "window_size": 80,
                        }
                    )
    for method, variant in [*methods, ("HIGH", "reference")]:
        name = f"{method}/{variant}" if variant not in {"classical", "reference"} else method
        for samples in (80, 200, 400, 800):
            for task in ("attitude", "strapdown"):
                value = samples / 80 * (2 if method != "identity" else 1)
                downstream.append(
                    {
                        "protocol": "penn_only",
                        "recording": "bs",
                        "task": task,
                        "method": name,
                        "samples": samples,
                        "horizon_s": (samples - 1) / 40,
                        "optimizer_runs": 3 if variant in {"default", "tuned"} else 1,
                        "windows_per_run": 8,
                        "endpoint_rmse_deg_mean": value if task == "attitude" else np.nan,
                        "endpoint_rmse_deg_sample_sd": 0.1,
                        "position_endpoint_rmse_m_mean": value if task == "strapdown" else np.nan,
                        "position_endpoint_rmse_m_sample_sd": 0.1,
                    }
                )
    signal = pd.DataFrame(
        {"sequence_id": np.repeat(["a", "b"], 800), "timestamp_s": np.tile(np.arange(800) / 40, 2)}
    )
    mapping = {}
    for index, axis in enumerate(["gx", "gy", "gz", "ax", "ay", "az"]):
        low = np.sin(signal.timestamp_s * (index + 1)) + 0.01 * rng.normal(size=len(signal))
        signal[axis + "_low"] = low
        signal[axis + "_high"] = low + 0.05
        signal[axis + "_virtual"] = low + 0.02
    mapping["virtual"] = {
        "method": "informer/tuned",
        "seed": 20261010,
        "family": "informer",
        "variant": "tuned",
    }
    return {
        "metrics": pd.DataFrame(metrics),
        "training": pd.DataFrame(training),
        "downstream": pd.DataFrame(downstream),
        "temporal": [
            {
                "protocol_id": "penn_only",
                "identity": "Penn_LOW_HIGH",
                "frame": signal,
                "methods": mapping,
                "source_tables": [],
            }
        ],
    }


def test_training_and_classical_coverage_keep_every_seed_and_negative_result():
    tool = module()
    pages = tool.plan_pages(fixture(), tool.default_config())
    train = [page for page in pages if page["kind"] == "training"]
    assert len(train) == 2
    assert all(page["coverage"]["final_run_count"] == 6 for page in train)
    accuracy = next(page for page in pages if page["kind"] == "metrics")
    assert accuracy["data"]["model_family"].nunique() == 11
    assert len(accuracy["data"]) == 13 * 6
    downstream = [page for page in pages if page["kind"] == "downstream"]
    assert len(downstream) == 2
    assert any(
        (page["data"].get("gain_percent", pd.Series(dtype=float)) < 0).any() for page in downstream
    )


def test_representative_segment_rule_cannot_see_reference_or_prediction_errors():
    tool = module()
    data = fixture()["temporal"][0]["frame"]
    first = tool.select_low_excitation_segment(data, window_points=800)
    changed = data.copy()
    for name in changed:
        if name.endswith(("_high", "_virtual")):
            changed[name] = np.random.default_rng(4).normal(size=len(changed)) * 100000
    assert tool.select_low_excitation_segment(changed, window_points=800) == first
    bad = data.copy()
    bad.loc[100, "timestamp_s"] += 0.1
    with pytest.raises(ValueError, match="contiguous"):
        tool.select_low_excitation_segment(bad, window_points=800)


def test_synthetic_exports_and_validated_resume_are_explicitly_nonfinal(tmp_path):
    tool = module()
    tables = fixture()
    pages = tool.plan_pages(tables, tool.default_config())
    chosen = []
    for kind in ("metrics", "training", "downstream", "signals", "spectral"):
        chosen.append(next(page["id"] for page in pages if page["kind"] == kind))
    index = tool.build_synthetic(tables, tmp_path / "figures", page_ids=chosen)
    assert index["synthetic"] and index["synthetic_fixture"]
    assert index["qa_summary"]["visual_review_status"] == "pending"
    assert len(index["figures"]) == 5
    for figure in index["figures"]:
        assert set(figure["files"]) == {"svg", "pdf", "png", "tiff"}
        assert figure["qa"]["status"] == "passed"
        assert figure["coverage"]
        for path in figure["files"].values():
            assert (tmp_path / "figures" / path).is_file()
    png = tmp_path / "figures" / index["figures"][0]["files"]["png"]
    stamp = png.stat().st_mtime_ns
    tool.build_synthetic(tables, tmp_path / "figures", page_ids=chosen, resume=True)
    assert png.stat().st_mtime_ns == stamp
    png.write_bytes(png.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        tool.build_synthetic(tables, tmp_path / "figures", page_ids=chosen, resume=True)


def test_formal_cli_has_no_synthetic_bypass_and_rejects_pending_package(tmp_path):
    tool = module()
    package = tmp_path / "package/metadata"
    package.mkdir(parents=True)
    (package / "package_manifest.json").write_text(
        json.dumps({"schema_version": 1, "status": "complete", "synthetic": True})
    )
    with pytest.raises(ValueError, match="Synthetic"):
        tool.main(["--results-root", str(package.parent), "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def extended_fixture():
    tables = fixture()
    neural = (
        tables["metrics"].query("method == 'model' and variant == 'tuned'").drop(columns="variant")
    )
    tables["deltas"] = neural.rename(columns={"mean": "delta_mean", "sample_sd": "delta_sample_sd"})
    tables["deltas"]["delta_mean"] = np.tile([-0.1, 0.05], len(neural) // 2)
    tables["transfer"] = tables["metrics"].assign(protocol_id="alimu_to_penn", reference_mean=0.05)
    tables["transfer"].loc[tables["transfer"].method.ne("model"), "reference_mean"] = np.nan
    ranks = []
    for protocol in ("penn_only", "pooled"):
        for variant in ("default", "tuned"):
            for rank, family in enumerate(("transformer", "informer"), 1):
                ranks.append(
                    {
                        "task": "paired",
                        "split": "test",
                        "aggregation_level": "pair",
                        "metric": "gyro_vector_rmse_radps",
                        "unit": "rad/s",
                        "model_family": family,
                        "protocol_id": protocol,
                        "dataset_id": "penncosyvio",
                        "variant": variant,
                        "low_sensor": "LOW",
                        "high_sensor": "HIGH",
                        "rank": rank,
                    }
                )
    tables["ranks"] = pd.DataFrame(ranks)
    tables["rank_correlations"] = pd.DataFrame(
        [
            {
                "task": "paired",
                "split": "test",
                "metric": "gyro_vector_rmse_radps",
                "unit": "rad/s",
                "variant": "tuned",
                "comparison": "different_training_and_evaluation_datasets",
                "aggregation_level": "pair",
                "left_protocol": "alimu_only",
                "left_dataset": "alimu",
                "left_low_sensor": "LOW_A",
                "left_high_sensor": "HIGH_A",
                "right_protocol": "penn_only",
                "right_dataset": "penncosyvio",
                "right_low_sensor": "LOW_B",
                "right_high_sensor": "HIGH_B",
                "spearman_rho": -1.0,
                "kendall_tau_b": -1.0,
            }
        ]
    )
    latency = [
        {
            "protocol_id": "penn_only",
            "pair_identity": "all channels",
            "task": "paired",
            "device": "cpu",
            "batch_size": 1,
            "window_size": 80,
            "model_family": family,
            "variant": variant,
            "parameters": 1000,
            "wall_p50_ms": 0.1,
            "wall_p95_ms": 0.2,
            "throughput": 10000,
            "measurement_id": family + variant,
            "precision": "FP32",
        }
        for family in ("transformer", "informer")
        for variant in ("default", "tuned")
    ]
    tables["latency_neural"] = pd.DataFrame(latency)
    tables["latency_classical"] = pd.DataFrame(
        [
            {
                **latency[0],
                "model_family": method,
                "variant": "classical",
                "fitted_scalar_count": 0 if method == "identity" else 42,
                "measurement_id": method,
                "precision": "float64",
            }
            for method in ("identity", "affine")
        ]
    )
    return tables


def test_every_additional_figure_branch_preserves_negative_data_and_draws():
    tool = module()
    pages = tool.plan_pages(extended_fixture(), tool.default_config())
    assert {"deltas", "transfer", "ranks", "correlations", "latency", "segment_integrals"} <= {
        page["kind"] for page in pages
    }
    for page in pages:
        figure = tool.draw_page(page, tool.default_config())
        figure.canvas.draw()
        tool.plt.close(figure)
    delta = next(page for page in pages if page["kind"] == "deltas")
    assert delta["data"]["mean"].min() < 0 < delta["data"]["mean"].max()


def test_latency_reads_actual_environment_and_keeps_fixed_seed_architecture_mapping():
    tool = module()
    entries = [
        {
            "entry_id": str(index),
            "seed": [20261010, 20261011, 20261012][index % 3],
            "protocol_id": "penn_only",
            "model_family": "transformer",
            "variant": "default" if index % 6 < 3 else "tuned",
            "task": "paired",
            "architecture_id": "shared",
            "timing_representative_entry_id": "0",
        }
        for index in range(558)
    ]
    report = {
        "status": "complete",
        "entries": 558,
        "records": [
            {
                "architecture_id": "shared",
                "seed": 20261010,
                "wall": {"count": 300, "p50_ms": 0.1, "p95_ms": 0.2},
                "device": "cuda:0",
                "batch_size": 1,
                "window_size": 80,
                "parameters": 100,
                "parameter_bytes": 400,
                "loop_throughput_windows_per_second": 1000,
                "precision": "FP32",
            }
        ],
    }
    receipts = {
        "latency.json": report,
        "checkpoint_inventory.json": {"entries": entries},
        "environment.json": {"torch": "2.4.0", "device": "cuda:0", "gpu": "SYNTHETIC_DEVICE"},
    }

    class Reader:
        def json(self, path):
            return receipts[Path(path).name]

    result = tool._latency_table(Reader(), Path("receipt"))
    assert len(result) == 186
    assert result.measurement_id.nunique() == 1
    assert result.environment.str.contains("SYNTHETIC_DEVICE").all()
    assert not any("sample_sd" in name for name in result.columns)
    receipts["latency.json"]["records"][0].update(device="cpu", batch_size=128)
    receipts["environment.json"] = {
        "torch": "2.4.0",
        "device": "cpu",
        "cpu_processor": "SYNTHETIC_CPU",
    }
    with pytest.raises(ValueError, match="CPU B=1"):
        tool._latency_table(Reader(), Path("receipt"))
    receipts["environment.json"] = {}
    with pytest.raises(ValueError, match="hardware"):
        tool._latency_table(Reader(), Path("receipt"))


def test_completion_audit_requires_all_roles_and_finite_evidence_but_allows_unused_nan():
    tool = module()
    finals, jobs = [], {}
    for index in range(558):
        job = {
            "id": str(index),
            "status": "complete",
            "phase": "final_default" if index % 2 == 0 else "final_tuned",
            "checkpoint_sha256": "a" * 64,
            "run_dir": "runs/" + str(index),
            "evaluation_artifact_sha256": {
                "runs/" + str(index) + "/evaluation/" + split + "_predictions.parquet": "c" * 64
                for split in ("validation", "test")
            },
        }
        finals.append(job)
        jobs[job["id"]] = job
    audit = {
        "schema_version": 1,
        "status": "passed",
        "synthetic": False,
        "source_pipeline_sha256": "b" * 64,
        "counts": {
            "selected_cells": 93,
            "final_evaluated": 558,
            "transfer_models": 120,
            "downstream_views": 5,
        },
        "source_roster_check": {"status": "passed"},
        "unused_optional_nan": float("nan"),
        "checkpoint_checks": [
            {
                "job_id": job["id"],
                "phase": job["phase"],
                "source_path": job["run_dir"] + "/best.ckpt",
                "sha256": "a" * 64,
                "finite": True,
                "tensor_count": 1,
                "tensor_elements": 4,
            }
            for job in finals
        ],
        "prediction_checks": [
            {
                "job_id": job["id"],
                "split": split,
                "source_path": job["run_dir"] + "/evaluation/" + split + "_predictions.parquet",
                "sha256": "c" * 64,
                "finite": True,
                "identity_nonnull": True,
                "rows": 80,
                "columns": 10,
                "checked_numeric_columns": ["gx_low"],
                "checked_identity_columns": ["sequence_id"],
            }
            for job in finals
            for split in ("validation", "test")
        ],
        "selection_checks": [{} for _ in range(93)],
        "marker_checks": [
            {"status": "passed", "source_path": str(i) + ".log", "sha256": "d" * 64}
            for i in range(3)
        ],
    }

    class Package:
        def resolve_path(self, value):
            return Path(value)

    class Reader:
        package = Package()

        def json(self, _):
            return audit

        def path(self, path, expected):
            assert expected in {"a" * 64, "c" * 64, "d" * 64}
            return Path(path)

    assert tool.verify_completion_audit(Reader(), {"jobs": jobs}, finals, "b" * 64) is audit
    original_path = audit["prediction_checks"][0]["source_path"]
    audit["prediction_checks"][0]["source_path"] = audit["prediction_checks"][2]["source_path"]
    with pytest.raises(ValueError, match="exact job/split"):
        tool.verify_completion_audit(Reader(), {"jobs": jobs}, finals, "b" * 64)
    audit["prediction_checks"][0]["source_path"] = original_path
    audit["prediction_checks"][0]["finite"] = False
    with pytest.raises(ValueError, match="prediction finite"):
        tool.verify_completion_audit(Reader(), {"jobs": jobs}, finals, "b" * 64)
    audit["prediction_checks"][0]["finite"] = True
    audit["prediction_checks"].pop()
    with pytest.raises(ValueError, match="role"):
        tool.verify_completion_audit(Reader(), {"jobs": jobs}, finals, "b" * 64)


def test_transfer_duplicate_seed_cannot_look_like_three_seed_statistics():
    tool = module()
    records = []
    for variant in ("default", "tuned"):
        for seed in (20261010, 20261011, 20261012):
            metrics = {name: 1 + (seed - 20261010) * 0.1 for name in tool.PAIRED_METRICS}
            identity = {"dataset_id": "penncosyvio", "low_sensor": "LOW", "high_sensor": "HIGH"}
            records.append(
                {
                    "direction": "alimu_to_penn",
                    "model_family": "informer",
                    "variant": variant,
                    "seed": seed,
                    "target_fingerprint": "one_target",
                    "classical_state_sha256": "one_state",
                    "metrics": {
                        "pairs": [
                            {
                                **identity,
                                "metrics": {
                                    "model": metrics,
                                    "identity": {name: 2 for name in metrics},
                                },
                            }
                        ]
                    },
                    "target_domain_reference": {
                        "test": {
                            "pairs": [
                                {**identity, "metrics": {"model": {name: 0.5 for name in metrics}}}
                            ]
                        }
                    },
                }
            )
    result = tool._transfer_table(records)
    assert result.query("method == 'model'").n.eq(3).all()
    assert result.query("variant == 'classical'").n.eq(1).all()
    assert (
        result.query("method == 'model'")["mean"] > result.query("method == 'model'").reference_mean
    ).all()
    with pytest.raises(ValueError, match="duplicate"):
        tool._transfer_table([*records, records[0]])


def saved_result_package_fixture(root, tool):
    """Real small CSV/JSON/Parquet files; permanently synthetic, never weights."""
    roots = {
        "core_project": "server/core_project",
        "core_pipeline": "server/core_project/runs/toy_core",
        "aux1_project": "server/aux1",
        "aux1": "server/aux1/runs/toy_aux1",
        "aux2_project": "server/aux2",
        "aux2": "server/aux2/runs/toy_aux2",
    }
    mappings = [
        {"source_prefix": "/synthetic/" + name, "mirror_prefix": mirror}
        for name, mirror in (
            ("core_project", roots["core_project"]),
            ("aux1", roots["aux1_project"]),
            ("aux2", roots["aux2_project"]),
        )
    ]

    def original(member):
        for mapping in mappings:
            if member.startswith(mapping["mirror_prefix"] + "/"):
                return mapping["source_prefix"] + member[len(mapping["mirror_prefix"]) :]
        return None

    def put(member, value):
        path = root / member
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, pd.DataFrame):
            if path.suffix == ".parquet":
                value.to_parquet(path, index=False)
            else:
                value.to_csv(path, index=False, lineterminator="\n")
        else:
            path.write_text(json.dumps(value), encoding="utf-8")
        return path

    paired_protocols = [
        "pooled",
        "alimu_only",
        "penn_only",
        "penn_loso_af",
        "penn_loso_as",
        "penn_loso_bf",
    ]
    protocols = [(name, "paired") for name in paired_protocols] + [
        (f"navigation_rotation_{i}", "navigation") for i in range(3)
    ]
    families = [
        "tcn",
        "plain_cnn",
        "gru",
        "transformer",
        "cnn_bilstm",
        "deepgyro_adapter",
        "tlio_resnet18_adapter",
        "informer",
        "patchtst",
        "itransformer",
    ]
    state, finals, registry, metric_rows = {"cells": {}}, [], [], []
    t = np.arange(320) / 40
    physical = {
        "sequence_id": ["al_train_test_segment"] * 320,
        "timestamp_s": t,
        "dataset_id": ["alimu"] * 320,
        "low_sensor": ["LOW"] * 320,
        "high_sensor": ["HIGH"] * 320,
    }
    for i, axis in enumerate(tool.AXES):
        physical[axis + "_low"] = np.sin(t * (i + 1))
        physical[axis + "_high"] = np.sin(t * (i + 1)) + 0.05
    for protocol, task in protocols:
        selected_families = families + (
            ["tartanimu_window_adapter"] if task == "navigation" else []
        )
        methods = (
            tool.CLASSICAL if task == "paired" else ["zero", "train_mean", "ridge", "ridge_tuned"]
        )
        metrics = tool.PAIRED_METRICS if task == "paired" else tool.NAV_METRICS
        for family in selected_families:
            cell = protocol + "/" + family
            state["cells"][cell] = {
                "protocol": {"id": protocol, "task": task},
                "candidate": {"id": family},
            }
            for variant in ("default", "tuned"):
                for seed in (20261010, 20261011, 20261012):
                    job_id = protocol + "__" + family + "__" + variant + "__" + str(seed)
                    directory = roots["core_pipeline"] + "/experiments/" + job_id
                    config_member = roots["core_pipeline"] + "/configs/" + job_id + ".yaml"
                    config_path = put(
                        config_member,
                        {
                            "training": {
                                "checkpoint_metric": "val_score",
                                "epochs": 3,
                                "loss": "toy",
                            },
                            "data": {"window_size": 80},
                        },
                    )
                    job = {
                        "id": job_id,
                        "cell": cell,
                        "phase": "final_" + variant,
                        "seed": seed,
                        "run_dir": original(directory + "/placeholder").rsplit("/", 1)[0],
                        "config": original(config_member),
                        "config_sha256": tool.sha256_file(config_path),
                    }
                    finals.append(job)
                    registry.append({"job_id": job_id, "cell": cell, "seed": seed})
                    put(
                        directory + "/metrics.csv",
                        pd.DataFrame(
                            {
                                "epoch": [1, 2, 3],
                                "train_loss": [3.0, 2.0, 1.0],
                                "val_loss": [4.0, 3.0, 2.0],
                                "val_score": [1.5, 1.2, 1.1],
                            }
                        ),
                    )
                    if protocol == "alimu_only" and seed == 20261010:
                        pred = pd.DataFrame(physical)
                        for axis in tool.AXES:
                            pred["model_" + axis] = pred[axis + "_low"] + 0.02
                            for method in tool.CLASSICAL:
                                pred[method + "_" + axis] = pred[axis + "_low"] + (
                                    0 if method == "identity" else 0.03
                                )
                        put(directory + "/evaluation/test_predictions.parquet", pred)
                for metric in metrics:
                    metric_rows.append(
                        {
                            "task": task,
                            "protocol_id": protocol,
                            "split": "test",
                            "aggregation_level": "pair" if task == "paired" else "dataset",
                            "dataset_id": "alimu"
                            if protocol in {"alimu_only", "pooled"}
                            else "penncosyvio"
                            if task == "paired"
                            else "smartpnt",
                            "low_sensor": "LOW",
                            "high_sensor": "HIGH",
                            "method": "model",
                            "model_family": family,
                            "variant": variant,
                            "metric": metric,
                            "mean": 1.0,
                            "sample_sd": 0.1,
                            "n": 3,
                            "unit": "rad/s"
                            if metric.startswith("gyro")
                            else "m/s²"
                            if task == "paired"
                            else "m",
                        }
                    )
        for method in methods:
            for metric in metrics:
                metric_rows.append(
                    {
                        **metric_rows[-1],
                        "method": method,
                        "model_family": method,
                        "variant": "classical",
                        "metric": metric,
                        "mean": 1.2,
                        "sample_sd": np.nan,
                        "n": 1,
                        "unit": "rad/s"
                        if metric.startswith("gyro")
                        else "m/s²"
                        if task == "paired"
                        else "m",
                    }
                )
    aggregate = roots["aux2"] + "/accuracy_tables/"
    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame = pd.concat(
        [metrics_frame, metrics_frame.assign(split="validation")], ignore_index=True
    )
    put(aggregate + "metrics_seed_summary.csv", metrics_frame)
    delta = (
        metrics_frame.query("method == 'model' and variant == 'tuned'")
        .drop(columns="variant")
        .rename(columns={"mean": "delta_mean", "sample_sd": "delta_sample_sd"})
    )
    put(aggregate + "matched_seed_delta_summary.csv", delta)
    put(aggregate + "model_rankings.csv", metrics_frame.query("method == 'model'").assign(rank=1.0))
    put(aggregate + "rank_correlations.csv", extended_fixture()["rank_correlations"])
    put(aggregate + "run_registry.csv", pd.DataFrame(registry))
    transfer_records = []
    for direction in ("alimu_to_penn", "penn_to_alimu"):
        identity = {
            "dataset_id": "penncosyvio" if direction == "alimu_to_penn" else "alimu",
            "low_sensor": "LOW",
            "high_sensor": "HIGH",
        }
        for family in families:
            for variant in ("default", "tuned"):
                for seed in (20261010, 20261011, 20261012):
                    values = {metric: 1 + (seed - 20261010) * 0.1 for metric in tool.PAIRED_METRICS}
                    transfer_records.append(
                        {
                            "direction": direction,
                            "model_family": family,
                            "variant": variant,
                            "seed": seed,
                            "target_fingerprint": direction,
                            "classical_state_sha256": direction,
                            "metrics": {
                                "pairs": [
                                    {
                                        **identity,
                                        "metrics": {
                                            "model": values,
                                            **{
                                                method: {metric: 2.0 for metric in values}
                                                for method in tool.CLASSICAL
                                            },
                                        },
                                    }
                                ]
                            },
                            "target_domain_reference": {
                                "test": {
                                    "pairs": [
                                        {
                                            **identity,
                                            "metrics": {
                                                "model": {metric: 0.5 for metric in values}
                                            },
                                        }
                                    ]
                                }
                            },
                        }
                    )
    put(
        roots["aux1"] + "/transfer/transfer_metrics.json",
        {"status": "complete", "records": 120, "source_model_records": transfer_records},
    )
    summaries = []
    for protocol, recording in (
        ("penn_loso_af", "af"),
        ("penn_loso_as", "as"),
        ("penn_loso_bf", "bf"),
        ("penn_only", "bs"),
        ("pooled", "bs"),
    ):
        mapping, signals = (
            {
                "high": {
                    "method": "HIGH",
                    "seed": None,
                    "category": "reference",
                    "family": None,
                    "variant": None,
                }
            },
            dict(physical),
        )
        counter = 0
        for family in families:
            for variant in ("default", "tuned"):
                for seed in (20261010, 20261011, 20261012):
                    prefix = f"neural_{counter:03d}"
                    counter += 1
                    mapping[prefix] = {
                        "method": family + "/" + variant,
                        "family": family,
                        "variant": variant,
                        "seed": seed,
                    }
                    for axis in tool.AXES:
                        signals[axis + "_" + prefix] = physical[axis + "_low"] + 0.02
        for method in tool.CLASSICAL:
            mapping["classic_" + method] = {
                "method": method,
                "seed": None,
                "family": None,
                "variant": "classical",
            }
            for axis in tool.AXES:
                signals[axis + "_classic_" + method] = physical[axis + "_low"] + (
                    0 if method == "identity" else 0.03
                )
        unit = roots["aux1"] + "/downstream/units/" + protocol + "/"
        put(
            unit + "metrics.json",
            {
                "recording": recording,
                "methods": mapping,
                "neural_optimizer_runs": 60,
                "neural_method_variants": 20,
                "unique_classical_methods": 9,
            },
        )
        put(unit + "evaluated_signals.parquet", pd.DataFrame(signals))
        for method in [
            *[family + "/" + variant for family in families for variant in ("default", "tuned")],
            *tool.CLASSICAL,
            "HIGH",
        ]:
            for samples in (80, 200, 400, 800):
                for task in ("attitude", "strapdown"):
                    summaries.append(
                        {
                            "protocol": protocol,
                            "recording": recording,
                            "task": task,
                            "method": method,
                            "samples": samples,
                            "horizon_s": (samples - 1) / 40,
                            "optimizer_runs": 3 if "/" in method else 1,
                            "windows_per_run": 1,
                            "endpoint_rmse_deg_mean": samples / 80,
                            "endpoint_rmse_deg_sd": 0.1 if "/" in method else 0.0,
                            "position_endpoint_rmse_m_mean": samples / 40,
                            "position_endpoint_rmse_m_sd": 0.2 if "/" in method else 0.0,
                        }
                    )
    put(roots["aux1"] + "/downstream/overview/all_fold_summary.csv", pd.DataFrame(summaries))
    inventory = [
        {
            "entry_id": job["id"],
            "seed": job["seed"],
            "protocol_id": job["cell"].split("/")[0],
            "model_family": job["cell"].split("/")[1],
            "variant": job["phase"].removeprefix("final_"),
            "task": state["cells"][job["cell"]]["protocol"]["task"],
            "architecture_id": job["cell"].split("/")[1] + job["phase"],
            "timing_representative_entry_id": "fixed_seed",
        }
        for job in finals
    ]
    for name, device, batches in (("latency_cpu", "cpu", [1]), ("latency_gpu", "cuda:0", [1, 128])):
        records = [
            {
                "architecture_id": architecture,
                "seed": 20261010,
                "wall": {"count": 300, "p50_ms": 0.1, "p95_ms": 0.2},
                "device": device,
                "batch_size": batch,
                "window_size": 80,
                "parameters": 100,
                "parameter_bytes": 400,
                "loop_throughput_windows_per_second": 1000,
                "precision": "FP32",
            }
            for architecture in sorted({row["architecture_id"] for row in inventory})
            for batch in batches
        ]
        put(
            roots["aux1"] + "/" + name + "/latency.json",
            {"status": "complete", "entries": 558, "records": records},
        )
        put(roots["aux1"] + "/" + name + "/checkpoint_inventory.json", {"entries": inventory})
        put(
            roots["aux1"] + "/" + name + "/environment.json",
            {
                "torch": "2.4.0",
                "device": device,
                "gpu": "SYNTHETIC_GPU" if device != "cpu" else None,
                "cpu_processor": "SYNTHETIC_CPU",
            },
        )
    classic = [
        {
            "method": "identity",
            "task": "paired",
            "pair_identity": {"dataset_id": "alimu"},
            "contract": {"protocol_id": "alimu_only", "source_id": "synthetic"},
            "device": "cpu",
            "precision": "float64",
            "window_size": 80,
            "batch_size": 1,
            "fitted_scalar_count": 0,
            "wall": {"p50_ms": 0.001, "p95_ms": 0.002},
            "loop_throughput_windows_per_second": 100000,
        }
    ]
    put(
        roots["aux2"] + "/classical_latency/latency.json",
        {
            "status": "complete",
            "final_run_mappings": 558,
            "records": classic,
            "environment": {"processor": "SYNTHETIC_CPU"},
        },
    )
    files = {
        path.relative_to(root).as_posix(): tool.sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "synthetic": True,
        "roots": roots,
        "source_roots": {},
        "path_mappings": mappings,
        "files": files,
        "file_metadata": {
            member: {"size_bytes": (root / member).stat().st_size, "source_path": original(member)}
            for member in files
        },
    }
    put("metadata/package_manifest.json", manifest)
    return state, finals


def test_actual_saved_result_mirror_reader_covers_all_final_training_and_aux_schemas(tmp_path):
    tool = module()
    root = tmp_path / "saved_synthetic_package"
    root.mkdir()
    state, finals = saved_result_package_fixture(root, tool)
    package = tool.load_package(root, verify=True, allow_synthetic=True)
    reader = tool.SourceReader(package)
    roots = {
        "core": package.root_path("core_pipeline"),
        "aux1": package.root_path("aux1"),
        "aux2": package.root_path("aux2"),
    }
    tables = tool.read_saved_results(reader, roots, state, finals, tool.default_config())
    assert tables["training"].cell.nunique() == 93
    assert tables["training"].job_id.nunique() == 558
    assert len(tool.plan_pages({"training": tables["training"]}, tool.default_config())) == 93
    assert tables["transfer"].query("method == 'model'").n.eq(3).all()
    assert tables["transfer"].query("variant == 'classical'").n.eq(1).all()
    assert tables["downstream"].protocol.nunique() == 5
    with pytest.raises(ValueError, match="horizon"):
        tool.validate_downstream_coverage(tables["downstream"].query("samples != 800"), state)
    assert tables["latency_neural"].measurement_id.nunique() < len(tables["latency_neural"])
    assert tables["latency_neural"].environment.str.contains("SYNTHETIC_").all()
    assert len(tables["temporal"]) == 6
    assert reader.sources
    pages = tool.plan_pages(
        {"metrics": tables["metrics"], "downstream": tables["downstream"]}, tool.default_config()
    )
    metric_pages = [
        page
        for page in pages
        if page["kind"] == "metrics"
        and page["data"].protocol_id.eq("alimu_only").all()
        and page["data"].split.eq("test").all()
    ]
    assert len(metric_pages) == 3
    assert set().union(
        *(set(page["data"].query("method == 'model'").model_family) for page in metric_pages)
    ) == {cell.split("/", 1)[1] for cell in state["cells"] if cell.startswith("alimu_only/")}
    assert all(page["data"].apply(tool.label, axis=1).nunique() <= 17 for page in metric_pages)
    downstream_pages = [
        page
        for page in pages
        if page["kind"] == "downstream"
        and page["coverage"]["protocol"] == "penn_only"
        and not page["gains"]
    ]
    assert len(downstream_pages) == 3
    assert len(set().union(*(set(page["data"].method) for page in downstream_pages))) == 30
    assert all(set(page["data"].samples) == {80, 200, 400, 800} for page in downstream_pages)
    with pytest.raises(ValueError, match="Synthetic"):
        tool.load_package(root, verify=True)
