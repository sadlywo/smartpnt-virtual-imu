"""Build an immutable Chinese report from a complete reviewer artifact mirror.

No model loading, inference, fitting, dataset decoding or selection is performed.
Only the private fixture API accepts explicitly synthetic packages; the CLI
always requires real complete results and externally reviewed figure exports.
"""

# ruff: noqa: RUF001
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import platform
import re
import shutil
import statistics
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote

from reviewer_artifact_package import load_package, member_name, sha256_file, valid_sha
from reviewer_temporal_support import classify_temporal_support

SEEDS = [20261010, 20261011, 20261012]
VARIANTS = ["default", "tuned"]
FAMILIES = [
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
PROTOCOLS = [
    "pooled",
    "alimu_only",
    "penn_only",
    "penn_loso_af",
    "penn_loso_as",
    "penn_loso_bf",
    "navigation_rotation_0",
    "navigation_rotation_1",
    "navigation_rotation_2",
]
EXPECTED_CELLS = {
    p + "/" + f
    for p in PROTOCOLS
    for f in FAMILIES + (["tartanimu_window_adapter"] if p.startswith("navigation") else [])
}
CLASSICAL = [
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
DOWNSTREAM_VIEWS = {
    "penn_loso_af": "af",
    "penn_loso_as": "as",
    "penn_loso_bf": "bf",
    "penn_only": "bs",
    "pooled": "bs",
}
HORIZON_SAMPLES, HORIZONS = [80, 200, 400, 800], [1.975, 4.975, 9.975, 19.975]
PRIMARY = {
    "gyro_vector_rmse_radps": "rad/s",
    "accel_vector_rmse_mps2": "m/s^2",
    "vector_rmse_m": "m",
}
# Flattened keys from formal_evaluation.physical_error_metrics plus
# reviewer_evaluation._pair_metrics and the shared producer's two ratios/macro.
# Keep this explicit allowlist independent of the scientific and plotting runtimes.
SHARED_SIGNAL_UNITS = {
    "samples": "samples",
    "grouped_relative_score": "dimensionless",
    **{
        f"{sensor}_{metric}_{suffix}": unit
        for sensor, suffix, unit in (("gyro", "radps", "rad/s"), ("accel", "mps2", "m/s^2"))
        for metric in ("vector_rmse", "vector_mae", "bias_norm", "error_p95")
    },
    **{
        f"axis_{metric}_{axis}": unit
        for metric in ("rmse", "mae", "mean_error", "error_std")
        for axes, unit in ((("gx", "gy", "gz"), "rad/s"), (("ax", "ay", "az"), "m/s^2"))
        for axis in axes
    },
    **{
        metric + "_relative_identity": "dimensionless"
        for metric in ("gyro_vector_rmse_radps", "accel_vector_rmse_mps2")
    },
}
ACCURACY_TABLES = [
    "run_registry",
    "all_fit_jobs",
    "metrics_long",
    "metrics_seed_summary",
    "matched_seed_deltas",
    "matched_seed_delta_summary",
    "relative_identity_percent",
    "relative_identity_seed_summary",
    "model_rankings",
    "rank_correlations",
]
METRIC_GROUPS = [
    "task",
    "protocol_id",
    "training_scope",
    "split",
    "aggregation_level",
    "dataset_id",
    "low_sensor",
    "high_sensor",
    "sequence_id",
    "original_sequence_id",
    "physical_group_id",
    "method",
    "model_family",
    "variant",
    "metric",
    "unit",
]
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/reviewer_report.yaml"
TEMPORAL_HELPER = Path(__file__).with_name("reviewer_temporal_support.py")
SHARED_PROFILE = "reviewer_shared_classical_v1"
SHARED_SCOPE = "protocol_shared_classical"
METHOD_KEY = ("protocol", "information_scope", "canonical_method", "variant")
ORIGINAL_COUNTS = {
    "selected_cells": 93,
    "final_evaluated": 558,
    "transfer_models": 120,
    "downstream_views": 5,
}
SHARED_COUNTS = {
    "shared_protocols": 6,
    "shared_logical_controls": 54,
    "shared_learned_controls": 48,
    "shared_identity_controls": 6,
    "shared_signal_slots": 108,
    "shared_downstream_views": 5,
    "shared_downstream_summary_slots": 360,
    "shared_paired_final_mappings": 360,
}
EXPORT_SHARED_COUNTS = {**SHARED_COUNTS, "shared_paired_support_refs": 720}
SHARED_ROLES = {
    "control_registry",
    "candidate_history",
    "shared_state_inventory",
    "method_scope",
    "metrics_long",
    "signal_registry",
    "neural_support_registry",
    "alias_audit",
    "downstream_summary",
    "downstream_windows",
    "downstream_segments",
    "state_inventory",
    "bank_parameter_totals",
    "latency_summary",
    "latency_raw",
    "workload_latency",
}
SHARED_STAGES = ("fit_select", "signal_evaluation", "downstream", "cpu_latency")


def analysis_profile(pkg):
    profile = pkg.manifest.get("analysis_profile")
    require(profile in (None, "reviewer_original_v1", SHARED_PROFILE), "Unknown analysis profile")
    return profile or "reviewer_original_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label, *, nonnegative=False):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Missing/nonnumeric {label}: {value!r}") from error
    require(math.isfinite(result), f"nonfinite {label}")
    require(not nonnegative or result >= 0, f"Negative {label}")
    return result


def integer(value, label):
    number = finite(value, label, nonnegative=True)
    require(number == int(number), f"Noninteger {label}")
    return int(number)


def shared_metric_unit(metric):
    require(
        metric in SHARED_SIGNAL_UNITS, "Shared physical metric has no declared unit: " + str(metric)
    )
    return SHARED_SIGNAL_UNITS[metric]


def json_write(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def frozen_training_settings(pkg, final, cells):
    """Read flat scalar keys from our safe-dumped frozen configs, not general YAML.

    Nested maps/lists/tags are deliberately not interpreted. These explicitly
    required generated scalar fields have unambiguous JSON/YAML scalar syntax.
    """
    fields = {
        "model": {"type", "in_channels", "out_channels", "target_mode"},
        "training": {
            "epochs",
            "loss",
            "gyro_signal_weight",
            "accel_signal_weight",
            "derivative_weight",
            "spectral_weight",
            "preintegration_weight",
            "allan_weight",
            "lr",
            "weight_decay",
            "lr_scheduler",
            "warmup_epochs",
            "min_lr",
            "grad_clip_norm",
            "early_stopping_patience",
            "checkpoint_metric",
        },
    }
    rows = []
    for jid, job in sorted(final.items()):
        path = pkg.resolve(job["config"])
        pkg.verify_file(path, job["config_sha256"])
        text = path.read_text(encoding="utf-8-sig")
        try:
            source = json.loads(text)
        except json.JSONDecodeError:
            source, section = {"model": {}, "training": {}}, None
            for line in text.splitlines():
                top = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*):\s*", line)
                if top:
                    section = top.group(1)
                elif line and not line.startswith((" ", "#")):
                    section = None
                item = re.fullmatch(r"  ([A-Za-z_][A-Za-z_0-9]*):\s+([^#]+?)\s*", line)
                if item and section in fields and item.group(1) in fields[section]:
                    value = item.group(2)
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = value.strip("'\"")
                    require(
                        item.group(1) not in source[section], "Duplicate frozen training scalar"
                    )
                    source[section][item.group(1)] = value
        model, train = source["model"], source["training"]
        task = "navigation" if job["cell"].startswith("navigation") else "paired"
        cell = cells[job["cell"]]
        trial = cell["trials"][0] if job["phase"] == "final_default" else cell["selected"]["trial"]
        parameters = trial["parameters"]
        require(job["trial_id"] == trial["id"], "Frozen final trial identity differs")
        for key in ("lr", "weight_decay"):
            require(
                key in train and "training." + key in parameters,
                "Missing declared frozen optimizer scalar",
            )
            require(
                type(train[key]) in (int, float),
                "Frozen optimizer value must be a simple numeric scalar",
            )
            require(
                math.isclose(
                    finite(train[key], key, nonnegative=True),
                    finite(parameters["training." + key], "trial " + key, nonnegative=True),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ),
                "Frozen optimizer scalar differs from selected/default trial",
            )
        profile = parameters["loss_profile"]
        require(
            profile
            in (
                {"huber_only", "mse_only"}
                if task == "navigation"
                else {"huber_only", "mse_only", "composite"}
            ),
            "Invalid declared loss_profile",
        )
        require(
            train.get("loss") == ("mse" if profile == "mse_only" else "huber"),
            "Frozen loss differs from trial profile",
        )
        for key in fields["training"] - {"loss", "lr_scheduler", "checkpoint_metric"}:
            if key in train:
                require(
                    type(train[key]) in (int, float),
                    "Frozen training value must be a simple numeric scalar",
                )
                finite(train[key], "frozen " + key, nonnegative=True)
        require(
            model.get("in_channels") == 6
            and model.get("out_channels") == (3 if task == "navigation" else 6),
            "Frozen model channel contract differs",
        )
        require(
            train.get("epochs") == 80
            and train.get("lr_scheduler") == "cosine_warmup"
            and train.get("warmup_epochs") == 5
            and train.get("early_stopping_patience") == 15
            and train.get("grad_clip_norm") == 1.0
            and train.get("min_lr") == 1e-6,
            "Frozen reviewer training schedule differs",
        )
        require(
            train.get("checkpoint_metric")
            == ("val_vector_rmse" if task == "navigation" else "val_grouped_score"),
            "Frozen validation selection criterion differs",
        )
        require(train.get("loss") in {"huber", "mse"}, "Unsupported frozen loss")
        if task == "paired":
            require(
                model.get("target_mode") == "residual",
                "Paired model must predict normalized HIGH-LOW residual",
            )
            composite = profile == "composite"
            expected = {
                "gyro_signal_weight": 2.0 if composite else 1.0,
                "accel_signal_weight": 1.0,
                "derivative_weight": 0.02 if composite else 0.0,
                "spectral_weight": 0.02 if composite else 0.0,
                "preintegration_weight": 0.005 if composite else 0.0,
                "allan_weight": 0.0,
            }
            require(
                all(train.get(k) == v for k, v in expected.items()),
                "Frozen paired loss weights differ",
            )
        rows.append(
            {
                "job_id": jid,
                "cell": job["cell"],
                "variant": job["phase"].removeprefix("final_"),
                "seed": job["seed"],
                **{k: train.get(k, "") for k in sorted(fields["training"])},
                "in_channels": model["in_channels"],
                "out_channels": model["out_channels"],
                "target_mode": model.get("target_mode", "window_displacement"),
                "model_type": model.get("type", ""),
                **classify_temporal_support(task, job["cell"].split("/")[1], model.get("type")),
                "loss_profile": profile,
                "config_member": pkg.relative_path(path),
                "config_sha256": job["config_sha256"],
            }
        )
    return rows


def write_csv(path, rows):
    require(bool(rows), f"Refusing an empty required table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _stage_audit(pkg, key, filename, expected, core_sha):
    state = pkg.read_json(pkg.root_path(key) / filename)
    require(state.get("status") == "complete", f"Incomplete {key} workflow")
    require(state.get("core_pipeline_sha256") == core_sha, f"{key} refers to another core pipeline")
    require(set(state.get("stages", {})) == set(expected), f"Incomplete {key} stage coverage")
    for name, stage in state["stages"].items():
        require(
            stage.get("status") == "complete" and stage.get("artifact_sha256"),
            f"Incomplete stage {key}/{name}",
        )
        directory = pkg.root_path(key) / name
        observed = {
            member[len(pkg.relative_path(directory)) + 1 :]: digest
            for member, digest in pkg.files.items()
            if member.startswith(pkg.relative_path(directory) + "/")
        }
        require(observed == stage["artifact_sha256"], f"Stage inventory differs: {key}/{name}")
        for name, digest in stage["artifact_sha256"].items():
            pkg.verify_file(directory / member_name(name), digest)
    return state


def _core_audit(pkg):
    pipeline_path = pkg.root_path("core_pipeline") / "pipeline.json"
    state = pkg.read_json(pipeline_path)
    core_sha = pkg.verify_file(pipeline_path)
    require(
        state.get("status") == "complete" and not state.get("smoke"),
        "Formal report requires a complete nonsmoke core",
    )
    require(state.get("selection_uses_test") is False, "Test-driven selection is forbidden")
    require(
        set(state.get("cells", {})) == EXPECTED_CELLS,
        "Expected exactly 93 declared protocol/family selections",
    )
    require(
        all("selected" in cell for cell in state["cells"].values()), "Missing selected candidate"
    )
    require(
        state.get("final_seeds") == SEEDS and state.get("final_variants") == VARIANTS,
        "Final seed/variant contract differs",
    )
    final = {key: job for key, job in state["jobs"].items() if job["phase"].startswith("final_")}
    identities = [(j["cell"], j["phase"].removeprefix("final_"), j["seed"]) for j in final.values()]
    expected = {
        (cell, variant, seed) for cell in EXPECTED_CELLS for variant in VARIANTS for seed in SEEDS
    }
    require(
        len(final) == 558 and len(set(identities)) == 558 and set(identities) == expected,
        "Incomplete/duplicate 558-final identity matrix",
    )
    require(
        all(
            j.get("status") == "complete" and j.get("evaluation_status") == "complete"
            for j in final.values()
        ),
        "All final jobs must be evaluated",
    )
    config = state["contract"]["config"]
    search = config["search"]
    require(
        search.get("trials") == 24
        and search.get("rung_epochs") == [8, 24, 80]
        and search.get("rung_keep") == [24, 8, 2]
        and search.get("confirmation_candidates") == 2,
        "Frozen HPO budgets differ",
    )
    require(
        config.get("seed") == 20261024 and config.get("confirmation_seed") == 20261025,
        "Search/confirmation seed differs",
    )
    for path, digest in state["contract"]["source_sha256"].items():
        pkg.verify_file(path, digest)
    audit = pkg.read_json("metadata/completion_audit.json")
    require(
        audit.get("schema_version") == 1 and audit.get("status") == "passed",
        "Missing complete export scientific audit",
    )
    require(
        audit.get("source_pipeline_sha256") == core_sha
        and pkg.resolve(audit["source_pipeline"]) == pipeline_path,
        "Export audit core identity differs",
    )
    shared = analysis_profile(pkg) == SHARED_PROFILE
    require(
        not shared or audit.get("analysis_profile") == SHARED_PROFILE,
        "Shared export audit profile differs",
    )
    require(
        audit.get("counts") == {**ORIGINAL_COUNTS, **(EXPORT_SHARED_COUNTS if shared else {})},
        "Export audit has incomplete scientific scope",
    )
    require(
        audit.get("source_roster_check", {}).get("status") == "passed",
        "Frozen source roster audit missing",
    )
    require(
        bool(audit.get("synthetic"))
        == bool(pkg.manifest.get("synthetic") or pkg.manifest.get("synthetic_fixture")),
        "Synthetic export audit/package flags disagree",
    )
    checks = {row["job_id"]: row for row in audit.get("checkpoint_checks", [])}
    require(set(final) <= set(checks), "Checkpoint finite audit lacks final jobs")
    predictions = defaultdict(dict)
    for row in audit.get("prediction_checks", []):
        require(
            row["split"] not in predictions[row["job_id"]], "Duplicate prediction audit identity"
        )
        predictions[row["job_id"]][row["split"]] = row
    fits = {row["job_id"]: row for row in audit.get("fit_checks", [])}
    require(set(state["jobs"]) == set(fits), "Fit ledger audit omits completed or failed jobs")
    for jid, job in final.items():
        check = checks[jid]
        require(
            check.get("finite") is True and check["sha256"] == job["checkpoint_sha256"],
            "Final checkpoint finite/hash audit disagrees",
        )
        pkg.verify_file(check["source_path"], check["sha256"])
        require(
            set(predictions[jid]) == {"validation", "test"},
            "Final prediction audit requires both splits",
        )
        for row in predictions[jid].values():
            require(
                row.get("finite") is True
                and row.get("identity_nonnull") is True
                and integer(row["rows"], "prediction rows") > 0,
                "Incomplete finite prediction audit",
            )
            pkg.verify_file(row["source_path"], row["sha256"])
        fit = fits[jid]
        require(
            fit.get("status") == "passed"
            and fit.get("validation", {}).get("finite_and_selected_minimum") is True,
            "Final validation minimum not independently audited",
        )
        require(
            fit.get("evaluation", {}).get("status") == "passed"
            and fit["evaluation"]["metrics"].get("numeric_finite") is True,
            "Final metrics not audited",
        )
        for path, digest in fit["evaluation"]["artifact_sha256"].items():
            pkg.verify_file(path, digest)
        pkg.verify_file(fit["config_path"], fit["config_sha256"])
    selections = audit.get("selection_checks", [])
    require(
        len(selections) == 93 and {r["cell"] for r in selections} == EXPECTED_CELLS,
        "93 validation selection audits required",
    )
    for row in selections:
        require(row.get("status") == "passed", "Selection audit failed")
        pkg.verify_file(row["path"], row["sha256"])
    markers = audit.get("marker_checks", [])
    require(
        len(markers) == (4 if shared else 3) and all(r.get("status") == "passed" for r in markers),
        "Exact workflow terminal marker roster required (three original, plus aux3 for shared)",
    )
    marker_text = ""
    for row in markers:
        pkg.verify_file(row["source_path"], row["sha256"])
        marker_text += pkg.resolve(row["source_path"]).read_text(encoding="utf-8") + "\n"
    for marker in (
        "REVIEWER_CORE_PASSED",
        "WORKFLOW_EXIT=0",
        "REVIEWER_AUXILIARY_PASSED",
        "REVIEWER_FINAL_ANALYSIS_PASSED",
    ):
        require(marker in marker_text.splitlines(), f"Missing terminal marker {marker}")
    if shared:
        _, shared_marker = _shared_publication_state(pkg, pkg.root_path("aux3"), audit)
        require(shared_marker in marker_text.splitlines(), "Missing exact shared completion marker " + shared_marker)
    return state, final, core_sha, audit


def _accuracy_audit(pkg, final, core_sha):
    directory = pkg.root_path("aux2") / "accuracy_tables"
    summary = pkg.read_json(directory / "summary.json")
    require(
        summary.get("status") == "complete"
        and summary.get("runs") == 558
        and summary.get("source_pipeline_sha256") == core_sha
        and summary.get("no_inference_or_fitting") is True,
        "Accuracy aggregate completion/core identity differs",
    )
    tables = {name: pkg.read_csv(directory / (name + ".csv")) for name in ACCURACY_TABLES}
    registry = tables["run_registry"]
    require(
        len(registry) == 558 and {r["job_id"] for r in registry} == set(final),
        "558-run registry coverage differs",
    )
    for row in registry:
        job = final[row["job_id"]]
        require(
            row["protocol_id"] + "/" + row["model_family"] == job["cell"]
            and row["variant"] == job["phase"].removeprefix("final_")
            and integer(row["seed"], "seed") == job["seed"],
            "Run registry scientific identity differs",
        )
        require(
            row["checkpoint_sha256"] == job["checkpoint_sha256"]
            and row["config_sha256"] == job["config_sha256"],
            "Run registry hashes differ",
        )
        best, actual, budget = (
            integer(row[key], key) for key in ("best_epoch", "actual_epochs", "epoch_budget")
        )
        require(
            0 <= best <= actual <= budget and budget == 80, "Invalid best epoch/early-stop budget"
        )
        finite(row["validation_best_metric"], "validation score")
        finite(row["elapsed_seconds"], "fit elapsed", nonnegative=True)
    require(bool(tables["all_fit_jobs"]), "Complete failed/search/confirmation fit ledger missing")
    metrics = tables["metrics_seed_summary"]
    require(bool(metrics), "No seed summary metrics")
    verify_seed_summaries(tables["metrics_long"], metrics)
    observed = set()
    for row in metrics:
        metric = row["metric"]
        finite(row["mean"], metric, nonnegative=metric in PRIMARY)
        if row.get("sample_sd") not in (None, ""):
            finite(row["sample_sd"], "sample SD", nonnegative=True)
        n = integer(row["n"], "optimizer n")
        if row["method"] == "model":
            require(
                n == 3 and integer(row["optimization_seed_count"], "seed count") == 3,
                "Neural summary must use all three optimization seeds",
            )
            require(row.get("sample_sd") not in (None, ""), "Neural sample SD missing")
            if (
                row["split"] == "test"
                and row["aggregation_level"] == "dataset"
                and metric in PRIMARY
            ):
                require(
                    row["unit"] == PRIMARY[metric], "Metric unit differs from physical definition"
                )
                observed.add(
                    (row["protocol_id"] + "/" + row["model_family"], row["variant"], metric)
                )
        else:
            require(
                n == 1
                and integer(row["optimization_seed_count"], "classical seed count") <= 1
                and row.get("sample_sd") in (None, ""),
                "Deterministic classical rows must be deduplicated, without seed SD",
            )
    expected = {
        (cell, variant, metric)
        for cell in EXPECTED_CELLS
        for variant in VARIANTS
        for metric in (
            ["vector_rmse_m"]
            if cell.startswith("navigation")
            else ["gyro_vector_rmse_radps", "accel_vector_rmse_mps2"]
        )
    }
    require(expected <= observed, "Primary metrics lack a declared cell/variant")
    dedup = pkg.read_json(directory / "classical_deduplication.json")
    require(bool(dedup), "Full classical physical/output/metric fingerprint audit missing")
    keys = []
    for row in dedup:
        for field in ("physical_rows_sha256", "predictions_sha256", "saved_metrics_sha256"):
            valid_sha(row[field])
        keys.append(tuple(row[k] for k in ("task", "protocol_id", "split", "method")))
        require(
            bool(row.get("source_job_ids"))
            and set(row["source_job_ids"])
            == {
                jid for jid, job in final.items() if job["cell"].split("/")[0] == row["protocol_id"]
            },
            "Classical dedup source jobs missing",
        )
    require(len(keys) == len(set(keys)), "Classical repeated deterministic slot")
    expected_classical = {
        ("navigation" if p.startswith("navigation") else "paired", p, split, method)
        for p in PROTOCOLS
        for split in ("validation", "test")
        for method in (
            ["zero", "train_mean", "ridge", "ridge_tuned"]
            if p.startswith("navigation")
            else CLASSICAL
        )
    }
    require(
        set(keys) == expected_classical,
        "Complete 132 classical protocol/split/method fingerprints required",
    )
    return summary, tables


def verify_seed_summaries(long_rows, summaries):
    """Recompute saved scalar seed summaries, preserving every physical/unit key."""
    grouped = defaultdict(list)
    for row in long_rows:
        require(
            set([*METRIC_GROUPS, "seed", "value"]) <= set(row),
            "Saved long metrics schema incomplete",
        )
        key = tuple(row[k] for k in METRIC_GROUPS)
        grouped[key].append(row)
    seen = set()
    for summary in summaries:
        key = tuple(summary[k] for k in METRIC_GROUPS)
        require(
            key not in seen and key in grouped, "Duplicate/missing summary physical metric group"
        )
        seen.add(key)
        rows = grouped[key]
        values = [finite(r["value"], "saved seed metric") for r in rows]
        if summary["method"] == "model":
            require(
                sorted(integer(r["seed"], "metric seed") for r in rows) == SEEDS,
                "Saved neural metrics lack exact three seeds",
            )
        else:
            require(
                len(rows) == 1 and rows[0]["seed"] in (None, ""),
                "Classical repeats must be deduplicated before summarizing",
            )
        require(
            integer(summary["n"], "summary n") == len(values),
            "Summary n disagrees with saved metrics",
        )
        require(
            math.isclose(
                finite(summary["mean"], "summary mean"),
                statistics.mean(values),
                rel_tol=1e-9,
                abs_tol=1e-11,
            ),
            "Saved summary mean disagrees with seed metrics",
        )
        if len(values) > 1:
            require(
                math.isclose(
                    finite(summary["sample_sd"], "summary sample SD"),
                    statistics.stdev(values),
                    rel_tol=1e-9,
                    abs_tol=1e-11,
                ),
                "Saved sample SD must use ddof=1",
            )
        else:
            require(
                summary["sample_sd"] in (None, ""), "Deterministic classical SD must remain blank"
            )
    require(seen == set(grouped), "Saved metrics are omitted from the seed summary")


def _transfer_audit(pkg):
    base = pkg.root_path("aux1") / "transfer"
    value = pkg.read_json(base / "transfer_metrics.json")
    records = value.get("source_model_records", [])
    expected = {
        (d, f, v, s)
        for d in ("alimu_to_penn", "penn_to_alimu")
        for f in FAMILIES
        for v in VARIANTS
        for s in SEEDS
    }
    keys = [(r["direction"], r["model_family"], r["variant"], r["seed"]) for r in records]
    require(
        value.get("status") == "complete"
        and value.get("records") == 120
        and len(keys) == 120
        and set(keys) == expected,
        "120 zero-shot transfer identities required",
    )
    for row in records:
        require(
            row.get("selection_uses_target_labels") is False
            and not row.get("nonfinal_debug_subset"),
            "Transfer target labels used for selection/debug subset",
        )
        pkg.verify_file(row["checkpoint"], row["checkpoint_sha256"])
    rows = pkg.read_csv(base / "transfer_seed_summary.csv")
    require(bool(rows), "Transfer metric summary missing")
    for row in rows:
        require(
            integer(row["seeds"], "transfer seeds") == 3, "Transfer summaries must retain all seeds"
        )
        for key, val in row.items():
            if (key.endswith("_mean") or key.endswith("_sample_sd")) and val not in (None, ""):
                finite(val, "transfer " + key)
    return rows


def downstream_comparison(rows):
    """Compare units separately; pooled bs never enters the four-recording group."""
    lookup = {
        (r["protocol"], r["task"], integer(r["samples"], "samples"), r["method"]): r for r in rows
    }
    require(len(lookup) == len(rows), "Duplicate downstream method/task/horizon")
    result = []
    for key, row in lookup.items():
        protocol, task, samples, method = key
        if method == "identity":
            continue
        metric, unit = (
            ("endpoint_rmse_deg_mean", "deg")
            if task == "attitude"
            else ("position_endpoint_rmse_m_mean", "m")
        )
        value = finite(row[metric], "downstream " + metric, nonnegative=True)
        low = finite(
            lookup[(protocol, task, samples, "identity")][metric],
            "LOW downstream",
            nonnegative=True,
        )
        result.append(
            {
                "protocol": protocol,
                "recording": DOWNSTREAM_VIEWS[protocol],
                "view": "same_recording_pooled_reference"
                if protocol == "pooled"
                else "outer_recording_holdout",
                "task": task,
                "method": method,
                "horizon_s": finite(row["horizon_s"], "horizon"),
                "samples": samples,
                "metric": metric,
                "unit": unit,
                "mean": value,
                "low_mean": low,
                "optimizer_runs": integer(row["optimizer_runs"], "downstream seeds"),
                "sample_sd": row.get(metric.replace("_mean", "_sd"), ""),
                "improvement_over_low_pct": 100 * (1 - value / low) if low > 1e-12 else "",
                "behavior": "improved" if value < low else "degraded" if value > low else "equal",
            }
        )
    return result


def _downstream_audit(pkg):
    base = pkg.root_path("aux1") / "downstream"
    require(
        pkg.read_json(base / "complete.json").get("status") == "complete",
        "Downstream stage incomplete",
    )
    all_rows, units = [], []
    expected_methods = {f + "/" + v for f in FAMILIES for v in VARIANTS} | set(CLASSICAL) | {"HIGH"}
    for protocol, recording in DOWNSTREAM_VIEWS.items():
        directory = base / "units" / protocol
        value = pkg.read_json(directory / "metrics.json")
        require(
            value.get("protocol") == protocol and value.get("recording") == recording,
            "Downstream recording/fold identity differs",
        )
        require(
            value.get("view")
            == (
                "separate_same_recording_pooled_reference"
                if protocol == "pooled"
                else "outer_recording_holdout"
            ),
            "Pooled and Penn outer views must remain separate",
        )
        require(
            [
                value.get(k)
                for k in (
                    "neural_optimizer_runs",
                    "neural_method_variants",
                    "unique_classical_methods",
                    "method_variants_total",
                    "stream_count",
                )
            ]
            == [60, 20, 9, 30, 70],
            "Downstream requires 70 streams/30 method variants per view",
        )
        methods = list(value["methods"].values())
        require(
            len(methods) == 70 and {r["method"] for r in methods} == expected_methods,
            "Downstream method inventory differs",
        )
        neural_keys = {(r["method"], r["seed"]) for r in methods if r.get("category") == "neural"}
        require(
            neural_keys == {(f + "/" + v, s) for f in FAMILIES for v in VARIANTS for s in SEEDS},
            "Downstream seed coverage differs",
        )
        require(
            integer(value["physical_support"]["rows"], "downstream physical support") > 0,
            "Downstream support empty",
        )
        valid_sha(value["physical_support"]["physical_sha256"])
        finite(value["physical_support"]["source_clock_origin_s"], "fixed source clock origin")
        pkg.verify_file(value["independent_gt_path"], value["independent_gt_sha256"])
        rows = pkg.read_csv(directory / "summary.csv")
        keys = [(r["task"], r["method"], integer(r["samples"], "horizon samples")) for r in rows]
        require(
            len(keys) == 240
            and len(set(keys)) == 240
            and set(keys)
            == {
                (t, m, s)
                for t in ("attitude", "strapdown")
                for m in expected_methods
                for s in HORIZON_SAMPLES
            },
            "Downstream all methods/all four horizons required",
        )
        for row in rows:
            require(
                row["protocol"] == protocol and row["recording"] == recording,
                "Downstream summary recording differs",
            )
            require(
                math.isclose(
                    finite(row["horizon_s"], "actual horizon"),
                    (integer(row["samples"], "samples") - 1) / 40,
                    abs_tol=1e-9,
                ),
                "Downstream duration differs from 40Hz endpoint span",
            )
            require(
                integer(row["optimizer_runs"], "downstream optimizer count")
                == (3 if "/" in row["method"] else 1),
                "Repeated classical seeds must not inflate n",
            )
            require(
                integer(row["windows_per_run"], "supported window count") > 0,
                "No independently supported GT windows",
            )
            metric = (
                "endpoint_rmse_deg_mean"
                if row["task"] == "attitude"
                else "position_endpoint_rmse_m_mean"
            )
            finite(row[metric], metric, nonnegative=True)
            finite(row[metric.replace("_mean", "_sd")], "downstream sample SD", nonnegative=True)
        for table in ("trials", "window_details", "full_segments"):
            require(
                bool(pkg.read_csv(directory / (table + ".csv"))),
                "Downstream full support/window/segment diagnostics missing",
            )
        all_rows.extend(rows)
        units.append(value)
    return all_rows, units


def neural_timing_batches(source_name, settings):
    """Exact frozen aux1 protocol: CPU B=1, CUDA:0 B=1/128."""
    require(source_name in {"latency_cpu", "latency_gpu"}, "Unknown neural timing source")
    batches = [1] if source_name == "latency_cpu" else [1, 128]
    device = "cpu" if source_name == "latency_cpu" else "cuda:0"
    require(
        settings.get("device") == device and settings.get("batch_sizes") == batches,
        "Frozen neural device/batch protocol differs (CPU B=1; CUDA:0 B=1/128)",
    )
    require(
        settings.get("cpu_threads") == (1 if source_name == "latency_cpu" else 4)
        and settings.get("input_seed") == 20261024
        and settings.get("end_to_end") is False,
        "Frozen neural thread/input-seed/network-only settings differ",
    )
    require(
        [settings.get(k) for k in ("warmup", "iterations", "repeats")] == [30, 100, 3],
        "Frozen neural timing budget differs",
    )
    return batches


def _latency_audit(pkg, final, core_sha):
    output, inventory_ids = {}, set(final)
    arch_sets = []
    for name in ("latency_cpu", "latency_gpu"):
        base = pkg.root_path("aux1") / name
        value = pkg.read_json(base / "latency.json")
        require(
            value.get("status") == "complete"
            and value.get("entries") == 558
            and value.get("source_pipeline_sha256") == core_sha,
            "Neural latency coverage/core identity incomplete",
        )
        rows = pkg.read_csv(base / "checkpoint_inventory.csv")
        require(
            len(rows) == 558 and {r["entry_id"] for r in rows} == inventory_ids,
            "All 558 parameter inventory identities required",
        )
        for row in rows:
            require(
                row["checkpoint_sha256"] == final[row["entry_id"]]["checkpoint_sha256"],
                "Parameter checkpoint identity differs",
            )
            require(
                integer(row["input_channels"], "IMU channels") == 6
                and integer(row["parameters"], "parameters") > 0,
                "Neural six-channel/parameter inventory invalid",
            )
        architectures = {r["architecture_id"] for r in rows}
        arch_sets.append(
            {
                r["entry_id"]: (r["architecture_id"], r["parameters"], r["checkpoint_sha256"])
                for r in rows
            }
        )
        records = value.get("records", [])
        settings = value["timing_protocol"]
        batches = neural_timing_batches(name, settings)
        keys = [(r["architecture_id"], r["batch_size"]) for r in records]
        require(
            len(keys) == len(batches) * len(architectures)
            and set(keys) == {(a, b) for a in architectures for b in batches},
            "Neural timing architecture/batch coverage incomplete",
        )
        for row in records:
            require(
                row.get("device") == settings["device"],
                "Neural timing record device differs from its source settings",
            )
            require(
                row.get("seed") == 20261010 and row["representative_entry_id"] in final,
                "Timing must use predefined representative seed",
            )
            require(
                integer(row["wall"]["count"], "timing calls") == 300,
                "Timing repetitions must retain 300 calls",
            )
            p50 = finite(row["wall"]["p50_ms"], "timing p50", nonnegative=True)
            p95 = finite(row["wall"]["p95_ms"], "timing p95", nonnegative=True)
            require(
                p95 >= p50 and finite(row["loop_throughput_windows_per_second"], "throughput") > 0,
                "Invalid latency quantiles/throughput",
            )
        require(
            [settings.get(k) for k in ("warmup", "iterations", "repeats")] == [30, 100, 3],
            "Timing budget differs",
        )
        output[name] = {"inventory": rows, "report": value}
        output[name]["mapping"] = pkg.read_csv(base / "architecture_mapping.csv")
        output[name].update(
            source_member=pkg.relative_path(base / "latency.json"),
            source_sha256=pkg.verify_file(base / "latency.json"),
        )
    require(arch_sets[0] == arch_sets[1], "CPU/GPU checkpoint/architecture inventories differ")
    base = pkg.root_path("aux2") / "classical_latency"
    value = pkg.read_json(base / "latency.json")
    require(
        value.get("status") == "complete"
        and value.get("final_run_mappings") == 558
        and value.get("records"),
        "Classical latency complete scope missing",
    )
    mapping = pkg.read_csv(base / "final_run_mapping.csv")
    id_field = "entry_id" if "entry_id" in mapping[0] else "job_id"
    require(
        len(mapping) == 558 and {r[id_field] for r in mapping} == inventory_ids,
        "Classical latency 558 mapping differs",
    )
    output["classical_latency"] = {
        "report": value,
        "rows": pkg.read_csv(base / "classical_latency.csv"),
    }
    for row in output["classical_latency"]["rows"]:
        for key, val in row.items():
            if key.endswith("p50_ms") or key.endswith("p95_ms"):
                finite(val, key, nonnegative=True)
    return output


def _shared_digest(value):
    """Producer canonical JSON digest; independent of report's legacy digest."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _unique_rows(rows, fields, expected, label):
    keys = [tuple(row[field] for field in fields) for row in rows]
    require(
        len(keys) == len(set(keys)) and set(keys) == expected,
        "Shared " + label + " exact tuple coverage differs",
    )


def _boolean(value):
    require(value in (True, False, "True", "False", "true", "false"), "Invalid serialized boolean")
    return value in (True, "True", "true")


def _deterministic(row):
    require(
        integer(row.get("n"), "shared n") == 1 and row.get("sd") in (None, ""),
        "Shared classical n=1 and blank SD required",
    )


def _shared_publication_state(pkg, base, audit):
    """Resolve the explicit V8 publication without relabelling original exit1."""
    mode = audit.get("completion_mode")
    if mode is None:
        require(pkg.manifest.get("completion_mode") is None, "Package/publication mode differs")
        return pkg.read_json(base / "shared_controls_state.json"), "REVIEWER_SHARED_CONTROLS_PASSED"
    require(mode == "saved_stages_finalization_v1" and pkg.manifest.get("completion_mode") == mode,
            "Unknown shared publication completion mode")
    validation = audit.get("shared_execution_profile", {})
    selected = audit.get("selected_publication_state")
    require(validation.get("status") == "passed" and validation.get("completion_mode") == mode
            and validation.get("selected_publication_state") == selected
            and pkg.manifest.get("selected_publication_state") == selected,
            "Main manifest/audit selected publication identity differs")

    def read(ref, expected=None):
        require(isinstance(ref, dict) and {"path", "bytes", "sha256"} <= set(ref),
                "Publication reference is incomplete")
        path = pkg.resolve(ref["path"])
        require(expected is None or path == expected, "Publication reference resolves to a different role")
        pkg.verify_file(path, ref["sha256"])
        require(path.stat().st_size == ref["bytes"], "Publication reference byte size differs")
        return pkg.read_json(path)

    state = read(selected, base / "shared_controls_finalized_state.json")
    require(state.get("kind") == "reviewer_shared_finalized_state_v1"
            and state.get("completion_mode") == mode and state.get("status") == "complete",
            "Selected sibling is not the closed finalized state")
    root = base / "finalization"
    failed_ref = validation.get("preserved_failed_execution")
    require(failed_ref == audit.get("preserved_failed_execution")
            == state.get("finalization", {}).get("original_failed_state")
            and failed_ref.get("sha256") == "51c2aa095b286956df9edaed4b61aa123278fdc981c92ec8c14b2e51f330aa7b",
            "Preserved original failed state identity differs")
    failed = read(failed_ref, base / "shared_controls_state.json")
    require(failed.get("status") == "failed" and failed.get("error_type") == "KeyError"
            and failed.get("error") == "'seed'", "Original failure was changed or relabelled success")
    for key in ("contract", "stages", "execution_profile", "stage_executions", "imported_completed_stages",
                "global_selection_set_sha256", "core_pipeline_sha256"):
        require(state.get(key) == failed.get(key), "Publication changed original scientific field: " + key)
    original = validation.get("original_scientific_execution", {})
    original_profile = original.get("profile", {})
    require(original_profile == validation.get("profile") == state.get("execution_profile")
            and original_profile.get("sha256") == "090ed6a023c5b0fbe11cc0fbf9013d3ff4714a4e27c9518a6902eb61e66dba03",
            "Original scientific execution profile differs")
    read(original_profile, base / "execution_profile/execution_profile.json")
    old_exit = read(original.get("parent_execution", {}).get("contained_control_receipts", {})
                    .get("evaluation_exit_receipt.json"), base / "execution_profile/evaluation_exit_receipt.json")
    require(old_exit.get("child_started") is True and old_exit.get("child_exit_code") == 1
            and old_exit.get("supervisor_error") is None, "Original parent exit1 is required")
    execution = validation.get("finalization_execution", {})
    require(execution == audit.get("finalization_execution"), "Finalizer execution descriptor differs")
    profile_ref, plan_ref, proof_ref = (validation.get(key) for key in
                                      ("finalization_profile", "publication_plan", "validation_proof"))
    require(profile_ref == state.get("finalization_profile") == execution.get("profile")
            and plan_ref == state["finalization"].get("publication_plan") == execution.get("publication_plan")
            and proof_ref == state["finalization"].get("validation_proof") == execution.get("validation_proof"),
            "Finalization profile/plan/proof references disagree")
    profile = read(profile_ref, root / "finalization_profile.json")
    require(profile.get("kind") == "shared_saved_stage_finalization_v1"
            and profile.get("completion_mode") == mode
            and all(profile.get(k) is False for k in ("allow_fitting", "allow_evaluation", "allow_timing"))
            and profile.get("original", {}).get("state") == failed_ref
            and profile["original"].get("profile", {}).get("sha256") == original_profile["sha256"],
            "Finalizer profile does not preserve original failure or forbids no science")
    proof = read(proof_ref, root / "validation_proof.json")
    typed = proof.get("identity_metadata_proof", {})
    descriptor = typed.get("descriptor", {})
    require(typed.get("kind") == "typed_identity_metadata_validation_proof_v1"
            and typed.get("complete") is True and typed.get("group_count") == 20
            and typed.get("identity_pairs") == 1136 and typed.get("window_pairs") == 1126
            and typed.get("segment_pairs") == 10
            and all(typed.get(k) is True for k in ("both_decorator_domains_exact",
                "all_original_non_decorator_fields_compared", "no_row_or_metric_mutation"))
            and typed.get("seed_invented") is False
            and descriptor.get("coordinator_source", {}).get("sha256")
                == "b28525bdc787d29e862bf4d23538c2db0d6981144b8f23426a679b5aef3180e4"
            and descriptor.get("schema_evidence", {}).get("sha256")
                == "bdcb79026ef9b92845060e6c3eddacbe40ea3c4d0d79bbdc91cc80668d6a47e4"
            and descriptor.get("overlay_source", {}).get("sha256")
                == "074852ef61e23a8486e1c4ff8f67dc0faa1d510bd245a7a6c98dfa4eae7ec060"
            and all(descriptor.get(k) is True for k in ("rest_function_AST_unchanged",
                "other_coordinator_bindings_unchanged", "full_original_function_normal_return_required"))
            and proof.get("source_checks") and proof.get("input_checks"),
            "Typed full-validator source/schema/coverage proof differs")
    index_path = base / "shared_controls_artifacts.json"
    index = pkg.read_json(index_path)
    index_sha = pkg.verify_file(index_path)
    require(proof.get("index") == index and state.get("shared_controls_artifacts_sha256") == index_sha,
            "Full saved validation proof/index differs")
    plan = read(plan_ref, root / "publication_plan.json")
    expected = dict(index_sha256=index_sha, identity_metadata_proof_sha256=hashlib.sha256(
        (json.dumps(typed, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()).hexdigest())
    require(plan.get("kind") == "shared_saved_stage_publication_plan_v1"
            and plan.get("completion_mode") == mode and plan.get("expected") == expected,
            "Independent publication plan differs from saved full proof")
    exited = read(execution.get("exit_receipt"), root / "finalization_exit_receipt.json")
    require(exited.get("child_started") is True and exited.get("child_exit_code") == 0
            and exited.get("child_birth") == execution.get("child")
            and exited.get("supervisor_error") is None and exited.get("no_retry_or_signal") is True
            and exited.get("next_queue_started") is False and exited.get("full_study_complete") is False,
            "Independent finalizer exit0/birth is missing")
    console_ref = execution.get("console")
    require(execution.get("marker") == "REVIEWER_SHARED_FINALIZATION_PASSED", "Finalizer marker differs")
    console_path = pkg.resolve(console_ref["path"])
    require(console_path == root / "finalization_console.log", "Finalizer console role differs")
    pkg.verify_file(console_path, console_ref["sha256"])
    lines = [line.strip() for line in console_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(lines and lines[-1] == execution["marker"], "Finalizer terminal marker missing")
    retired = read(execution.get("retirement"), root / "finalization_retirement.json")
    require(retired.get("profile") == profile_ref and retired.get("publication_plan") == plan_ref
            and retired.get("state") == selected and retired.get("index", {}).get("sha256") == index_sha
            and retired.get("exit") == execution["exit_receipt"] and retired.get("console") == console_ref
            and retired.get("child_birth") == execution["child"]
            and retired.get("supervisor_birth") == execution["supervisor"]
            and retired.get("child_retired") is True and retired.get("supervisor_retired") is True,
            "Finalizer retirement does not bind selected publication")
    return state, execution["marker"]


def _shared_audit(pkg, final, core_sha):
    """Verify immutable public interface without importing training/science runtimes."""
    require(
        {"aux3_project", "aux3"} <= set(pkg.roots),
        "Shared profile requires fourth project/run roots; original package insufficient",
    )
    base = pkg.root_path("aux3")

    def path(name):
        target = base / member_name(name)
        require(target.resolve().is_relative_to(base), "Shared artifact escapes aux3 run")
        return target

    index_path = base / "shared_controls_artifacts.json"
    index = pkg.read_json(index_path)
    index_sha = pkg.verify_file(index_path)
    export = pkg.read_json("metadata/completion_audit.json")
    reference = export.get("shared_controls_index", {})
    require(
        reference.get("sha256") == index_sha
        and pkg.resolve(reference["source_path"]) == index_path
        and export.get("auxiliary_checks", {}).get("aux3") == index,
        "Export shared index mapping/SHA/full audit differs from actual aux3 index",
    )
    require(
        index.get("schema_version") == 1
        and index.get("status") == "complete"
        and index.get("analysis_profile") == SHARED_PROFILE
        and index.get("counts") == SHARED_COUNTS,
        "Shared public index profile/counts incomplete",
    )
    original = index.get("original", {})
    require(
        original.get("core_pipeline_sha256") == core_sha
        and original.get("selected_cells") == 93
        and original.get("evaluated_final_records") == 558
        and original.get("neural_transfer_records") == 120
        and original.get("downstream_views") == 5,
        "Shared index original 93/558/120/5 identity differs",
    )
    pkg.verify_file(
        path(original["original_full_audit_path"]), original["original_full_audit_sha256"]
    )
    freeze_sha = valid_sha(index["global_selection_set_sha256"])
    pkg.verify_file(path("fit_select/selection_set.json"), freeze_sha)
    freeze = pkg.read_json(path("fit_select/selection_set.json"))
    protocols = set(PROTOCOLS[:6])
    require(
        freeze.get("status") == "frozen_before_any_test_load"
        and freeze.get("analysis_profile") == SHARED_PROFILE
        and set(freeze.get("protocols", {})) == protocols,
        "All six protocol selections must freeze before any TEST",
    )
    fit_contract = pkg.read_json(path("fit_select/contract.json"))
    fit_sha = _shared_digest(fit_contract)
    require(
        index.get("fit_contract_sha256") == fit_sha
        and freeze.get("contract_sha256") == fit_sha
        and fit_contract.get("pipeline_sha256") == core_sha,
        "Shared fit/source/core contract differs",
    )
    # Source/config/data manifests remain explicit links in source_index, never guessed.
    pkg.verify_file(fit_contract["config_path"], fit_contract["config_sha256"])
    require(
        bool(fit_contract.get("source_sha256")) and bool(fit_contract.get("manifests")),
        "Shared source and physical split/strict manifest references missing",
    )
    for source, digest in fit_contract["source_sha256"].items():
        pkg.verify_file(source, digest)
    for references in fit_contract["manifests"].values():
        for reference in references.values():
            source = reference["path"]
            if not str(source).startswith("/") and not re.match(r"^[A-Za-z]:[/\\]", str(source)):
                source = pkg.root_path("core_project") / member_name(source)
            pkg.verify_file(source, reference["sha256"])
    for protocol, descriptor in freeze["protocols"].items():
        pkg.verify_file(path(descriptor["path"]), descriptor["sha256"])
        selection = pkg.read_json(path(descriptor["path"]))
        require(
            selection.get("protocol") == protocol
            and selection.get("contract_sha256") == fit_sha
            and selection.get("source_sha256") == fit_contract["source_sha256"]
            and selection.get("manifests") == fit_contract["manifests"][protocol],
            "Shared per-protocol selection/source/split contract differs",
        )
    manifests = index.get("stage_manifests", {})
    require(set(manifests) == set(SHARED_STAGES), "Shared four stage manifests required")
    descriptors = {}
    for stage, reference in manifests.items():
        require(
            reference["path"] == stage + "/artifact_manifest.json"
            and reference["summary_path"] == stage + "/stage_summary.json",
            "Shared stage reference does not match its stage",
        )
        pkg.verify_file(path(reference["path"]), reference["sha256"])
        pkg.verify_file(path(reference["summary_path"]), reference["summary_sha256"])
        manifest = pkg.read_json(path(reference["path"]))
        summary = pkg.read_json(path(reference["summary_path"]))
        require(
            manifest.get("status") == "complete"
            and manifest.get("analysis_profile") == SHARED_PROFILE
            and manifest.get("root_kind") == "aux3_run"
            and summary.get("status") == "complete"
            and summary.get("stage") == stage
            and summary.get("analysis_profile") == SHARED_PROFILE
            and summary.get("selection_set_sha256") == freeze_sha
            and manifest.get("contract_sha256") == summary.get("contract_sha256"),
            "Shared stage profile/selection/contract differs",
        )
        expected = {
            name[len(pkg.relative_path(base)) + 1 :]: digest
            for name, digest in pkg.files.items()
            if name.startswith(pkg.relative_path(base / stage) + "/")
            and name != pkg.relative_path(path(reference["path"]))
        }
        require(manifest.get("files") == expected, "Shared complete stage inventory differs")
        for name, digest in manifest["files"].items():
            require(name.startswith(stage + "/"), "Shared stage member has wrong root")
            pkg.verify_file(path(name), digest)
        if stage in ("fit_select", "signal_evaluation"):
            require(summary["contract_sha256"] == fit_sha, "Shared fit/signal contract differs")
        else:
            contract = pkg.read_json(path(stage + "/contract.json"))
            require(
                _shared_digest(contract) == summary["contract_sha256"],
                "Shared downstream/CPU contract differs",
            )
            if stage == "downstream":
                require(
                    contract.get("fit_selection_set_sha256") == freeze_sha
                    and contract.get("signal_manifest_sha256")
                    == manifests["signal_evaluation"]["sha256"],
                    "Shared downstream dependency DAG differs",
                )
            else:
                require(
                    contract.get("selection_set_sha256") == freeze_sha
                    and contract.get("fit_contract_sha256") == fit_sha
                    and contract.get("pipeline_sha256") == core_sha
                    and contract.get("upstream_manifests")
                    == {name: manifests[name]["sha256"] for name in SHARED_STAGES[:3]},
                    "Shared CPU dependency DAG differs",
                )
        for role, descriptor in summary.get("tables", {}).items():
            require(role not in descriptors, "Shared table role ambiguous")
            descriptors[role] = descriptor
    require(
        descriptors == index.get("tables") and SHARED_ROLES <= set(descriptors),
        "Shared named public table roles incomplete/detached",
    )
    rows = {}
    for role, descriptor in descriptors.items():
        target = path(descriptor["path"])
        pkg.verify_file(target, descriptor["sha256"])
        table = pkg.read_csv(target)
        with target.open(encoding="utf-8-sig", newline="") as stream:
            columns = next(csv.reader(stream))
        require(
            len(table) == descriptor["rows"] and columns == descriptor["columns"],
            "Shared descriptor row/column inventory differs: " + role,
        )
        for row in table:
            require(
                not any(
                    str(value).lower() in {"nan", "inf", "+inf", "-inf", "infinity"}
                    for value in row.values()
                ),
                "Shared public table contains nonfinite value",
            )
        rows[role] = table
    logical = {(p, SHARED_SCOPE, m, "shared") for p in protocols for m in CLASSICAL}
    for role in ("control_registry", "shared_state_inventory", "method_scope", "alias_audit"):
        _unique_rows(rows[role], METHOD_KEY, logical, role)
        for row in rows[role]:
            _deterministic(row)
    for row in rows["method_scope"]:
        require(
            not _boolean(row["identity_routes_state"])
            and row["input_scope"] == "six_LOW_and_same_stream_LOW_history",
            "Shared prediction must use one state and LOW-only history",
        )
    for row in rows["control_registry"]:
        identity = row["canonical_method"] == "identity"
        require(
            _boolean(row["identity_alias"]) == identity
            and _boolean(row["learned_state"]) != identity
            and row["normalization"] == "none_raw_si"
            and integer(row["normalizer_count"], "normalizer") == 0,
            "Shared identity/learned/normalization contract differs",
        )
        pkg.verify_file(path(row["state_path"]), row["state_sha256"])
        for name in ("train_row_sha256", "validation_row_sha256", "numeric_sha256"):
            valid_sha(row[name])
        count = integer(row["coefficient_count"], "stored coefficients")
        method = row["canonical_method"]
        taps = integer(row["taps"], "taps") if row["taps"] else None
        expected_count = (
            0
            if identity
            else 6
            if method == "bias_mean"
            else 36 * (taps + int(method.startswith("arx"))) + 6
            if taps
            else 42
        )
        require(
            count == expected_count
            and integer(row["coefficient_bytes"], "coefficient bytes") == 8 * count,
            "Shared actual stored coefficient count/bytes differs",
        )
        state_path = path(row["state_path"])
        state = pkg.read_json(state_path)
        require(
            tuple(state.get(k) for k in METHOD_KEY) == tuple(row[k] for k in METHOD_KEY)
            and state.get("schema_version") == 1
            and state.get("input_channels") == state.get("output_channels") == 6,
            "Shared actual state identity/channel contract differs",
        )
        kind = "identity" if identity else "bias" if method == "bias_mean" else "linear"
        weights, intercept = state.get("weights"), state.get("intercept")
        require(
            isinstance(weights, list) and isinstance(intercept, list),
            "Shared actual coefficient arrays missing",
        )
        feature_count = 6 * ((taps or 1) + int(method.startswith("arx")))
        require(
            (kind == "identity" and weights == intercept == [])
            or (kind == "bias" and not weights and len(intercept) == 6)
            or (
                kind == "linear"
                and len(weights) == feature_count
                and all(isinstance(w, list) and len(w) == 6 for w in weights)
                and len(intercept) == 6
            ),
            "Shared coefficient array shape differs",
        )
        actual_count = len(intercept) + sum(len(w) for w in weights)
        for value in [v for w in weights for v in w] + intercept:
            finite(value, "actual learned coefficient")
        require(
            actual_count == count == state.get("coefficient_count")
            and state.get("coefficient_bytes") == actual_count * 8
            and state.get("normalizer_count") == state.get("normalizer_bytes") == 0
            and state.get("normalization") == {"mode": "none_raw_si", "coefficient_count": 0},
            "Shared coefficient/normalizer inventory disagrees with actual arrays",
        )
        require(
            state.get("state_kind") == kind
            and state.get("feature_spec")
            == {
                "taps": taps,
                "include_delta": method.startswith("arx"),
                "edge_padding": "first_sample",
                "initial_delta": 0,
                "split_local": True,
                "stream_key": "sequence_id",
            }
            and state.get("residual_add_low") == (kind == "bias"),
            "Shared history/LOW-only/normalization predictor contract differs",
        )
        alpha = finite(row["alpha"], "state alpha") if row["alpha"] else None
        require(
            state.get("hyperparameters") == {"alpha": alpha, "taps": taps},
            "Shared registry/state hyperparameters differ",
        )
        if method in {"identity", "bias_mean", "affine"}:
            require(alpha is None and taps is None, "Fixed identity/bias/OLS parameters differ")
        elif method in {"ridge_affine", "learned_fir_causal", "arx_linear"}:
            require(
                alpha == 0.01
                and taps
                == {"ridge_affine": None, "learned_fir_causal": 9, "arx_linear": 12}[method],
                "Fixed classical parameters differ",
            )
        numeric_fields = (
            "state_kind",
            "weights",
            "intercept",
            "hyperparameters",
            "feature_spec",
            "residual_add_low",
            "normalization",
            "arithmetic",
        )
        require(
            _shared_digest({k: state[k] for k in numeric_fields})
            == state.get("numeric_sha256")
            == row["numeric_sha256"],
            "Actual numeric state content SHA differs",
        )
        require(
            integer(row["serialized_state_bytes"], "serialized state bytes")
            == state_path.stat().st_size,
            "Serialized state bytes disagree with actual artifact",
        )
    split_keys = {(*key, split) for key in logical for split in ("validation", "test")}
    _unique_rows(rows["signal_registry"], (*METHOD_KEY, "split"), split_keys, "108 signals")
    signal_support = {}
    for row in rows["signal_registry"]:
        _deterministic(row)
        require(
            row["selection_set_sha256"] == freeze_sha and integer(row["rows"], "signal rows") > 0,
            "Shared signal selection/support differs",
        )
        key = (row["protocol"], row["split"])
        support = (row["rows"], valid_sha(row["row_sha256"]))
        require(
            key not in signal_support or signal_support[key] == support,
            "Shared methods have different physical support",
        )
        signal_support[key] = support
        pkg.verify_file(path(row["prediction_path"]))
    paired = {jid: job for jid, job in final.items() if not job["cell"].startswith("navigation")}
    neural = rows["neural_support_registry"]
    _unique_rows(
        neural,
        ("job_id", "split"),
        {(jid, split) for jid in paired for split in ("validation", "test")},
        "360/720 neural support",
    )
    neural_support = {}
    for row in neural:
        job = paired[row["job_id"]]
        require(
            row["protocol"] == job["cell"].split("/")[0]
            and row["cell"] == job["cell"]
            and row["phase"] == job["phase"]
            and integer(row["seed"], "support seed") == job["seed"],
            "Shared support roster scientific identity differs",
        )
        key = (row["protocol"], row["split"])
        require(row["rows"] == signal_support[key][0], "Shared/neural physical row count differs")
        # These two producers hash different column projections. Compare the
        # neural physical projection across every job, never to whole-frame SHA.
        support = (
            row["rows"],
            valid_sha(row["physical_sha256"]),
            row.get("physical_columns_json", ""),
        )
        require(
            key not in neural_support or neural_support[key] == support,
            "All neural jobs must have identical physical support/column projection",
        )
        neural_support[key] = support
        pkg.verify_file(row["prediction_path"], row["prediction_sha256"])
    require(
        {tuple(row[k] for k in (*METHOD_KEY, "split")) for row in rows["metrics_long"]}
        == split_keys,
        "Shared metric logical coverage differs",
    )
    for row in rows["metrics_long"]:
        _deterministic(row)
        finite(row["value"], "shared metric")
        shared_metric_unit(row["metric"])
    require(
        {
            (*tuple(row[k] for k in (*METHOD_KEY, "split")), row["metric"])
            for row in rows["metrics_long"]
            if row["metric"] in PRIMARY
        }
        >= {
            (*key, metric)
            for key in split_keys
            for metric in ("gyro_vector_rmse_radps", "accel_vector_rmse_mps2")
        },
        "Shared metrics lack gyro or acceleration primary slot",
    )
    # Candidate order and strict ties are checked on each protocol's own VAL.
    alphas = [1e-6, 1e-4, 0.01, 0.1, 1, 10, 100]
    require(
        len(rows["candidate_history"]) == 546, "Full six-protocol tuned candidate history required"
    )
    require(
        all(
            r["information_scope"] == SHARED_SCOPE and r["variant"] == "shared"
            for r in rows["candidate_history"]
        ),
        "Shared candidate information scope differs",
    )
    for protocol in protocols:
        for method, taps_grid in (
            ("ridge_tuned", [None]),
            ("fir_tuned", [1, 3, 5, 9, 17, 33, 65]),
            ("arx_tuned", [3, 6, 12, 24, 48]),
        ):
            candidates = [
                r
                for r in rows["candidate_history"]
                if r["protocol"] == protocol and r["canonical_method"] == method
            ]
            observed = [
                (
                    integer(r["taps"], "candidate taps") if r["taps"] else None,
                    finite(r["alpha"], "candidate alpha"),
                )
                for r in candidates
            ]
            require(
                observed == [(t, a) for t in taps_grid for a in alphas],
                "Shared full frozen candidate grid/order differs",
            )
            require(
                all(_boolean(r["finite"]) and not r.get("error") for r in candidates),
                "Shared candidate failure must block completion",
            )
            best = min(candidates, key=lambda r: finite(r["validation_score"], "candidate VAL"))
            selected = next(
                r
                for r in rows["control_registry"]
                if r["protocol"] == protocol and r["canonical_method"] == method
            )
            require(
                (
                    integer(selected["taps"], "selected taps") if selected["taps"] else None,
                    float(selected["alpha"]),
                )
                == (
                    integer(best["taps"], "best taps") if best["taps"] else None,
                    float(best["alpha"]),
                ),
                "Shared chosen hyperparameters do not reproduce strict-first VAL minimum",
            )
    for row in rows["alias_audit"]:
        status = row["alias_status"]
        require(
            status
            in {
                "exact_alias",
                "numerically_equivalent",
                "not_alias",
                "common_identity",
                "not_applicable_multi_pair_bank",
            },
            "Unknown shared alias status",
        )
        if status in {"exact_alias", "numerically_equivalent", "not_alias"}:
            for field in ("shared_state_sha256", "bank_state_sha256"):
                valid_sha(row[field])
            try:
                proof = json.loads(row["proof"])
            except json.JSONDecodeError:
                # Pandas serializes dict cells with Python repr; use a literal-only parser.
                import ast

                proof = ast.literal_eval(row["proof"])
            require(
                all(
                    proof.get(k) is True
                    for k in (
                        "same_protocol",
                        "train_rows_exact",
                        "validation_rows_exact",
                        "grid_order_exact",
                        "tie_rule_exact",
                        "bank_own_selection_valid",
                    )
                ),
                "Penn alias provenance/support/selection proof incomplete",
            )
            require(
                float(row["rtol"]) == 1e-10 and float(row["atol"]) == 1e-12,
                "Penn alias tolerance differs",
            )
        if row["protocol"] in protocols - {"pooled", "alimu_only"}:
            require(
                status in {"exact_alias", "numerically_equivalent", "not_alias"},
                "Penn single-pair alias requires actual proof",
            )
    expected_downstream = {
        (p, SHARED_SCOPE, m, "shared", task, str(samples))
        for p in DOWNSTREAM_VIEWS
        for m in CLASSICAL
        for task in ("attitude", "strapdown")
        for samples in HORIZON_SAMPLES
    }
    _unique_rows(
        rows["downstream_summary"],
        (*METHOD_KEY, "task", "samples"),
        expected_downstream,
        "360 downstream summary slots",
    )
    for row in rows["downstream_summary"]:
        _deterministic(row)
        require(
            row["recording"] == DOWNSTREAM_VIEWS[row["protocol"]]
            and integer(row["optimizer_runs"], "shared downstream runs") == 1,
            "Shared downstream recording/n differs",
        )
        require(
            finite(row["horizon_s"], "shared horizon") == (int(row["samples"]) - 1) / 40,
            "Shared fixed horizon differs",
        )
        for field in (
            "endpoint_rmse_deg",
            "position_endpoint_rmse_m",
            "horizontal_endpoint_rmse_m",
        ):
            if row.get(field) not in (None, ""):
                finite(row[field], field, nonnegative=True)
    require(
        rows["downstream_windows"] and rows["downstream_segments"],
        "Shared downstream window/full-segment evidence missing",
    )
    proofs = pkg.read_json(path("downstream/support_proofs.json"))
    require(
        len(proofs) == 10
        and {(r["protocol"], r["task"]) for r in proofs}
        == {(p, t) for p in DOWNSTREAM_VIEWS for t in ("attitude", "strapdown")}
        and all(r.get("identity_alias_verified") is True for r in proofs),
        "Shared independent GT/common-window identity proofs incomplete",
    )
    for row in rows["state_inventory"]:
        require(
            row["information_scope"] in (SHARED_SCOPE, "known_device_bank"),
            "State inventory scope differs",
        )
        require(
            integer(row["normalizer_count"], "normalizer count") == 0
            and integer(row["normalizer_bytes"], "normalizer bytes") == 0,
            "Raw SI classical fitted normalizer must be zero",
        )
    total_keys = {
        (p, s, m, v)
        for p in protocols
        for m in CLASSICAL
        for s, v in ((SHARED_SCOPE, "shared"), ("known_device_bank", "bank"))
    }
    _unique_rows(rows["bank_parameter_totals"], METHOD_KEY, total_keys, "state deployment totals")
    for row in rows["bank_parameter_totals"]:
        actual = [
            r
            for r in rows["state_inventory"]
            if tuple(r[k] for k in METHOD_KEY) == tuple(row[k] for k in METHOD_KEY)
        ]
        require(
            actual
            and integer(row["pair_count"], "deployment pair count") > 0
            and integer(row["bank_coefficient_count"], "deployment coefficients")
            == sum(integer(r["coefficient_count"], "state coefficients") for r in actual)
            and integer(row["bank_coefficient_bytes"], "deployment bytes")
            == sum(integer(r["coefficient_bytes"], "state bytes") for r in actual),
            "Whole-bank deployment totals disagree with unique per-pair state inventory",
        )
    for role in ("latency_summary", "workload_latency"):
        require(bool(rows[role]), "Shared actual CPU timing table missing")
        for row in rows[role]:
            _deterministic(row)
            require(
                row["device"] == "cpu"
                and row["dtype"] == "float64"
                and integer(row["warmup"], "warmup") == 30
                and integer(row["iterations"], "iterations") == 100
                and integer(row["repeats"], "repeats") == 3,
                "Shared actual CPU precision/budget differs",
            )
            for field in (
                "p50_ms",
                "p95_ms",
                "call_distribution_sd_ms",
                "loop_throughput_windows_per_second",
            ):
                finite(row[field], field, nonnegative=True)
            require(
                finite(row["p95_ms"], "P95") >= finite(row["p50_ms"], "P50"),
                "Measured P95 cannot be below P50",
            )
    workload_keys = {
        (*key, str(w), str(b))
        for key in {
            (p, s, m, v)
            for p in protocols
            for m in CLASSICAL
            for s, v in ((SHARED_SCOPE, "shared"), ("known_device_bank", "bank"))
        }
        for w in (80, 160, 320)
        for b in (1, 128)
    }
    _unique_rows(
        rows["workload_latency"],
        (*METHOD_KEY, "window_size", "batch_size_per_pair"),
        workload_keys,
        "matched workload latency",
    )
    _unique_rows(
        [r for r in rows["latency_summary"] if r["information_scope"] == SHARED_SCOPE],
        (*METHOD_KEY, "window_size", "batch_size_per_pair"),
        {key for key in workload_keys if key[1] == SHARED_SCOPE},
        "shared per-state latency",
    )
    workloads = {
        (
            r["protocol"],
            r["canonical_method"],
            r["window_size"],
            r["batch_size_per_pair"],
            r["information_scope"],
        ): r
        for r in rows["workload_latency"]
    }
    for (p, method, w, b, scope), row in workloads.items():
        other = workloads[
            (p, method, w, b, "known_device_bank" if scope == SHARED_SCOPE else SHARED_SCOPE)
        ]
        require(
            integer(row["pair_count"], "workload pairs") > 0
            and integer(row["windows_per_call"], "workload windows")
            == int(row["pair_count"]) * int(b)
            and row["pair_count"] == other["pair_count"]
            and row["windows_per_call"] == other["windows_per_call"],
            "Shared/bank whole-loop workload input counts differ",
        )
    state, _ = _shared_publication_state(pkg, base, export)
    require(
        state.get("status") == "complete"
        and state.get("analysis_profile") == SHARED_PROFILE
        and state.get("shared_controls_artifacts_sha256") == index_sha
        and state.get("counts") == SHARED_COUNTS
        and state.get("global_selection_set_sha256") == freeze_sha,
        "Shared completed queue state/index differs",
    )
    return {
        "index": index,
        "index_sha256": index_sha,
        "tables": rows,
        "descriptors": descriptors,
        "root": base,
    }


def _figures_audit(pkg, root, synthetic, shared=None):
    root = Path(root).resolve()
    path = root / "figure_index.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    require(
        index.get("schema_version") == 1 and index.get("status") == "complete",
        "Complete figure index required",
    )
    require(
        index.get("manifest_sha256") == pkg.manifest_sha256
        and index.get("source_manifest_sha256") == pkg.manifest_sha256,
        "Figures refer to another package manifest",
    )
    require(
        bool(index.get("synthetic")) == synthetic
        and bool(index.get("synthetic_fixture")) == synthetic,
        "Figure synthetic flags differ",
    )
    qa = index.get("qa_summary", {})
    require(
        qa.get("status") == "passed"
        and qa.get("pdf_render_checked") is True
        and qa.get("visual_review_status") == "passed",
        "Figure visual review/PDF render QA must pass",
    )
    require(
        index.get("coverage", {}).get("training_cells") == 93
        and index["coverage"].get("training_final_runs") == 558,
        "All 93 training-curve appendix cells must be covered",
    )
    source_files = index.get("source_files", {})
    for member, digest in source_files.items():
        pkg.verify_file(member, digest)
    verified = {"figure_index.json": sha256_file(path)}

    def check(name, digest):
        target = (root / member_name(name)).resolve()
        require(
            target.is_relative_to(root) and target.is_file(),
            "Figure artifact escapes root or missing",
        )
        require(sha256_file(target) == valid_sha(digest), f"Figure/QA checksum changed: {name}")
        verified[name] = digest
        return target

    helper_sha = sha256_file(TEMPORAL_HELPER)
    require(
        index.get("temporal_support_sha256") == helper_sha,
        "Figure temporal classifier missing/stale or differs from report helper",
    )
    temporal_source = index.get("temporal_support_source", {})
    require(
        temporal_source.get("sha256") == helper_sha,
        "Figure temporal helper source binding missing/different",
    )
    check(temporal_source["path"], helper_sha)

    cells = set()
    require(bool(index.get("figures")), "Figure inventory empty")
    for figure in index["figures"]:
        require(
            set(figure["files"]) == {"png", "pdf", "svg", "tiff"},
            "All four figure export formats required",
        )
        for fmt, name in figure["files"].items():
            check(name, figure["sha256"][fmt])
        value = figure["qa"]
        require(
            value.get("status") == "passed" and value.get("visual_review_status") == "passed",
            "Per-figure visual review must pass",
        )
        qa_file = check(value["path"], value["sha256"])
        qa_payload = json.loads(qa_file.read_text(encoding="utf-8"))
        require(
            qa_payload.get("status") == "passed",
            "Per-figure QA artifact failed",
        )
        for row in figure.get("source_tables", []):
            pkg.verify_file(row["path"], row["sha256"])
        for row in figure.get("source_data", []):
            check(row["path"], row["sha256"])
        coverage = figure.get("coverage", {})
        if coverage.get("group") != "training":
            require(
                qa_payload.get("temporal_support_sha256") == helper_sha,
                "Nontraining figure QA must bind current temporal classifier",
            )
        if coverage.get("group") == "training":
            require(
                coverage["cell"] not in cells
                and coverage.get("final_run_count") == 6
                and len(set(coverage.get("run_ids", []))) == 6,
                "Training appendix cell duplicated/incomplete",
            )
            cells.add(coverage["cell"])
    require(cells == EXPECTED_CELLS, "Training curve appendix cell identities differ")
    if shared:
        require(
            index.get("analysis_profile") == SHARED_PROFILE
            and index.get("shared_controls_index_sha256") == shared["index_sha256"]
            and index.get("shared_coverage")
            == {**SHARED_COUNTS, "neural_split_support_references": 720},
            "New shared profile requires its own figure coverage/index/QA; old QA insufficient",
        )
        shared_figures = [
            f
            for f in index["figures"]
            if f.get("coverage", {}).get("information_scope") == SHARED_SCOPE
        ]
        require(
            bool(shared_figures)
            and all(f.get("caption") and f.get("source_tables") for f in shared_figures),
            "Shared figures need explicit captions and aux3 sources",
        )
        for figure in shared_figures:
            require(
                any(
                    pkg.resolve(r["path"]).is_relative_to(shared["root"])
                    for r in figure["source_tables"]
                ),
                "Shared figure must link actual aux3 source table",
            )
            qa = json.loads((root / member_name(figure["qa"]["path"])).read_text(encoding="utf-8"))
            require(
                qa.get("analysis_profile") == SHARED_PROFILE
                and qa.get("shared_controls_index_sha256") == shared["index_sha256"],
                "Shared figure QA must bind the new profile and actual aux3 index",
            )
    return index, verified


def audit_inputs(results_root, figures_root, *, _allow_synthetic=False):
    pkg = load_package(results_root, verify=True, allow_synthetic=_allow_synthetic)
    synthetic = bool(pkg.manifest.get("synthetic") or pkg.manifest.get("synthetic_fixture"))
    state, final, core_sha, audit = _core_audit(pkg)
    _stage_audit(
        pkg,
        "aux1",
        "postprocessing_state.json",
        ["transfer", "downstream", "latency_cpu", "latency_gpu"],
        core_sha,
    )
    _stage_audit(
        pkg, "aux2", "final_analysis_state.json", ["accuracy_tables", "classical_latency"], core_sha
    )
    summary, tables = _accuracy_audit(pkg, final, core_sha)
    ledger = tables["all_fit_jobs"]
    require(
        len(ledger) == len(state["jobs"])
        and {row["id"] for row in ledger} == set(state["jobs"])
        and summary.get("fit_jobs") == len(ledger)
        and summary.get("failed_fit_jobs") == sum(row["status"] == "failed" for row in ledger),
        "Complete fit/failed ledger disagrees with core jobs",
    )
    transfer = _transfer_audit(pkg)
    downstream, units = _downstream_audit(pkg)
    latency = _latency_audit(pkg, final, core_sha)
    training_settings = frozen_training_settings(pkg, final, state["cells"])
    shared = (
        _shared_audit(pkg, final, core_sha) if analysis_profile(pkg) == SHARED_PROFILE else None
    )
    figures, figure_hashes = _figures_audit(pkg, figures_root, synthetic, shared)
    data = {
        "pkg": pkg,
        "pipeline": state,
        "final": final,
        "audit": audit,
        "core_sha256": core_sha,
        "summary": summary,
        "tables": tables,
        "registry": tables["run_registry"],
        "transfer": transfer,
        "downstream": downstream,
        "downstream_units": units,
        "latency": latency,
        "figures": figures,
        "figure_hashes": figure_hashes,
        "synthetic": synthetic,
        "training_settings": training_settings,
        "analysis_profile": analysis_profile(pkg),
        "shared": shared,
    }
    data["neural_parameters"], data["neural_latency_contexts"] = neural_cost_tables(data)
    data["neural_cost_display"] = neural_cost_display_rows(data["neural_latency_contexts"])
    return data


def _relative(path, base):
    target = Path(path).resolve()
    try:
        relative = os.path.relpath(target, Path(base).resolve())
    except ValueError:
        # Windows cannot express a relative link across drive letters.
        return target.as_uri()
    return quote(relative.replace(os.sep, "/"), safe="/._-~")


def scoped_signal_tables(data):
    """Keep every information condition in its key; common identity occurs once."""
    result, identities = [], {}
    dimensions = (
        "protocol",
        "split",
        "aggregation_level",
        "dataset_id",
        "low_sensor",
        "high_sensor",
        "sequence_id",
        "metric",
    )

    def _identity_key(row):
        # Only aggregate identity contexts have equivalent blank/sentinel spellings.
        level = row.get("aggregation_level", row.get("metric_scope", ""))
        fields = {k: row.get(k, "") for k in dimensions}
        fields["aggregation_level"] = level
        require(all(isinstance(v, str) for v in fields.values()), "Invalid identity context type")
        require(level in {"dataset", "pair", "sequence", "protocol_macro"}, "Unknown identity aggregation")
        if level == "dataset":
            require(
                (fields["low_sensor"], fields["high_sensor"], fields["sequence_id"])
                in {("", "", ""), ("__all_devices__", "__all_devices__", "__all_sequences__")},
                "Invalid dataset identity aggregate sentinel",
            )
            fields.update(low_sensor="", high_sensor="", sequence_id="")
        elif level == "pair":
            require(
                fields["low_sensor"] not in {"", "__all_devices__"}
                and fields["high_sensor"] not in {"", "__all_devices__"}
                and fields["sequence_id"] in {"", "__all_sequences__"},
                "Invalid pair identity aggregate context",
            )
            fields["sequence_id"] = ""
        elif level == "sequence":
            require(
                fields["low_sensor"] not in {"", "__all_devices__"}
                and fields["high_sensor"] not in {"", "__all_devices__"}
                and fields["sequence_id"] not in {"", "__all_sequences__"},
                "Invalid real sequence identity context",
            )
        else:
            require(
                not any(fields[k] for k in ("low_sensor", "high_sensor", "sequence_id")),
                "Invalid protocol macro identity context",
            )
        require(fields["protocol"] and fields["split"] and fields["metric"], "Incomplete identity context")
        require(level == "protocol_macro" or fields["dataset_id"], "Missing identity dataset")
        return tuple(fields[k] for k in dimensions)

    shared_identity_keys, shared_identity_metrics, shared_support = set(), set(), {}
    for original in data["shared"]["tables"]["metrics_long"]:
        if original["canonical_method"] != "identity":
            continue
        key = _identity_key(original)
        require(key not in shared_identity_keys, "Shared identity duplicate context")
        shared_identity_keys.add(key)
        shared_identity_metrics.add(original["metric"])
        if original["metric"] == "samples":
            samples = integer(original["value"], "shared identity support")
            require(samples > 0, "Shared identity support must be positive")
            shared_support[key[:-1]] = samples
    for original in data["tables"]["metrics_seed_summary"]:
        row = {
            **original,
            "protocol": original["protocol_id"],
            "canonical_method": original["model_family"]
            if original["method"] == "model"
            else original["method"],
            "information_scope": "unified_six_low_neural"
            if original["method"] == "model"
            else "navigation_baseline"
            if original["task"] == "navigation"
            else "known_device_bank",
            "variant": original["variant"] if original["method"] == "model" else "bank",
        }
        if original["task"] == "paired" and original["method"] == "identity":
            key = _identity_key(row)
            require(key not in identities, "Legacy identity duplicate context")
            finite(row["mean"], "legacy identity scalar")
            identities[key] = row
            continue
        result.append(row)
    common_identity_metrics = {row["metric"] for row in identities.values()} & shared_identity_metrics
    for original in data["shared"]["tables"]["metrics_long"]:
        row = {
            **original,
            "aggregation_level": original["metric_scope"],
            "unit": shared_metric_unit(original["metric"]),
            "mean": finite(original["value"], "shared scalar"),
            "sample_sd": "",
            "n": 1,
        }
        if row["canonical_method"] == "identity":
            key = _identity_key(row)
            require(
                row["metric"] not in common_identity_metrics or key in identities,
                "Common identity support context missing from legacy metrics",
            )
            if key in identities:
                require(identities[key]["unit"] == row["unit"], "Common identity metric unit differs")
                require(key[:-1] in shared_support, "Missing shared identity sample support")
                require(
                    integer(identities[key]["support_count"], "legacy identity support")
                    == shared_support[key[:-1]],
                    "Common identity sample support differs",
                )
                require(
                    math.isclose(
                        float(identities[key]["mean"]), row["mean"], rel_tol=1e-10, abs_tol=1e-12
                    ),
                    "Common shared/bank identity metric differs",
                )
                identities.pop(key)
            row.update(information_scope="common_identity", variant="reference")
        result.append(row)
    require(
        all(row["metric"] not in common_identity_metrics for row in identities.values()),
        "Common identity support context missing from shared metrics",
    )
    # Legacy metric definitions not emitted by aux3 remain traceable, also one identity.
    result.extend(
        {**row, "information_scope": "common_identity", "variant": "reference"}
        for row in identities.values()
    )
    require(
        len(
            {
                tuple(
                    row.get(k, "")
                    for k in (*dimensions, "information_scope", "canonical_method", "variant")
                )
                for row in result
            }
        )
        == len(result),
        "Scoped metric key collision",
    )
    main = [
        r
        for r in result
        if r["split"] == "test"
        and r["aggregation_level"] == "dataset"
        and r["metric"] in PRIMARY
        and r["information_scope"] != "navigation_baseline"
        and r["protocol"] in PROTOCOLS[:6]
    ]
    rankings = []
    for scope_set, label in (
        ({"unified_six_low_neural", SHARED_SCOPE, "common_identity"}, "main_common_information"),
        ({"known_device_bank", "common_identity"}, "supplementary_known_device_bank"),
    ):
        groups = defaultdict(list)
        for row in main:
            if row["information_scope"] in scope_set:
                groups[
                    (row["protocol"], row.get("dataset_id", ""), row["metric"], row["unit"])
                ].append(row)
        for group in groups.values():
            previous, rank = None, 0
            for position, row in enumerate(sorted(group, key=lambda r: float(r["mean"])), 1):
                value = float(row["mean"])
                if previous is None or value != previous:
                    rank = position
                previous = value
                rankings.append({**row, "ranking_condition": label, "rank": rank})
    return temporal_rows(result), temporal_rows(rankings)


def scoped_downstream_rows(data):
    rows, identity = [], {}
    for original in data["downstream"]:
        method = original["method"]
        scope = (
            "common_identity"
            if method == "identity"
            else "observed_high_reference"
            if method == "HIGH"
            else "unified_six_low_neural"
            if "/" in method
            else "known_device_bank"
        )
        canonical, variant = (
            method.split("/", 1)
            if "/" in method
            else (method, "reference" if method in {"identity", "HIGH"} else "bank")
        )
        task = original["task"]
        metric = "endpoint_rmse_deg" if task == "attitude" else "position_endpoint_rmse_m"
        row = {
            **original,
            "information_scope": scope,
            "canonical_method": canonical,
            "variant": variant,
            "mean": finite(original[metric + "_mean"], "downstream mean"),
            "sample_sd": original.get(metric + "_sd", "") if "/" in method else "",
            "n": 3 if "/" in method else 1,
            "metric": metric,
            "unit": "deg" if task == "attitude" else "m",
        }
        if method == "identity":
            identity[(row["protocol"], task, int(row["samples"]))] = row
        rows.append(row)
    for original in data["shared"]["tables"]["downstream_summary"]:
        metric = (
            "endpoint_rmse_deg" if original["task"] == "attitude" else "position_endpoint_rmse_m"
        )
        low = identity[(original["protocol"], original["task"], int(original["samples"]))]
        value = (
            finite(original[metric], "shared downstream mean")
            if original.get(metric) not in (None, "")
            else ""
        )
        if original["canonical_method"] == "identity":
            require(
                value != "" and math.isclose(value, low["mean"], rel_tol=1e-10, abs_tol=1e-12),
                "Common shared/bank downstream identity differs",
            )
            continue
        rows.append(
            {
                **original,
                "metric": metric,
                "unit": low["unit"],
                "mean": value,
                "sample_sd": "",
                "n": 1,
            }
        )
    for row in rows:
        low = identity[(row["protocol"], row["task"], int(row["samples"]))]["mean"]
        row["improvement_over_low_pct"] = (
            100 * (1 - row["mean"] / low) if row["mean"] != "" and low > 1e-12 else ""
        )
        row["behavior"] = (
            "unsupported"
            if row["mean"] == ""
            else "degraded"
            if row["mean"] > low
            else "improved"
            if row["mean"] < low
            else "equal"
        )
    return temporal_rows(rows, default_task="paired")


def temporal_row(row, *, default_task=None):
    """Project known report identities; no architecture/name heuristics select a method."""
    result = dict(row)
    protocol = row.get("protocol", row.get("protocol_id", ""))
    task = row.get("task")
    require(
        task in (None, "", "paired", "navigation", "attitude", "strapdown"),
        "Unknown temporal source task: " + str(task),
    )
    if task in {"attitude", "strapdown"}:
        task = "paired"
    if task not in {"paired", "navigation"}:
        task = default_task or (
            "navigation" if str(protocol).startswith("navigation") else "paired"
        )
    if str(protocol).startswith("navigation"):
        require(task == "navigation", "Temporal task and navigation protocol differ")
    elif protocol in PROTOCOLS[:6]:
        require(task == "paired", "Temporal task and paired protocol differ")
    method = row.get("canonical_method")
    if not method:
        method = row.get("method")
        if method == "model" or not method:
            method = row.get("model_family", row.get("family"))
        elif "/" in method:
            method = method.split("/", 1)[0]
    require(bool(method), "Temporal projection requires an authoritative method identity")
    if method == "HIGH":
        require(task == "paired", "HIGH reference is not a navigation estimator")
        fields = {
            "temporal_class": "measured_reference",
            "temporal_label": "Reference",
            "temporal_label_zh": "观测HIGH参考",
            "temporal_order": 3,
            "temporal_note": "Measured synchronous HIGH reference; not a learned LOW mapping or an end-to-end streaming guarantee.",
        }
    else:
        fields = classify_temporal_support(task, method, row.get("model_type") or None)
    return {**result, **fields}


def temporal_rows(rows, *, default_task=None):
    return [temporal_row(row, default_task=default_task) for row in rows]


def neural_cost_tables(data):
    """Bind every context to its declared measured representative, never arch alone."""
    registry = {r["job_id"]: r for r in data["registry"]}
    training_settings = {r["job_id"]: r for r in data["training_settings"]}
    parameters, latency = [], []
    for source_name in ("latency_cpu", "latency_gpu"):
        source = data["latency"][source_name]
        settings = source["report"]["timing_protocol"]
        batches = neural_timing_batches(source_name, settings)
        inventory = {r["entry_id"]: r for r in source["inventory"]}
        mapping_rows = source["mapping"]
        mapping = {r["entry_id"]: r for r in mapping_rows}
        require(
            len(mapping) == len(mapping_rows) and set(mapping) == set(registry) == set(inventory),
            "Cost join requires an unambiguous complete entry_id mapping",
        )
        arch_representatives, signatures = {}, {}
        for jid, inv in inventory.items():
            reg, mapped = registry[jid], mapping[jid]
            require(
                reg["task"]
                == ("navigation" if reg["protocol_id"].startswith("navigation") else "paired"),
                "Cost registry task differs from declared protocol",
            )
            identity = (
                reg["task"],
                reg["model_family"],
                int(inv["window_size"]),
                int(inv["parameters"]),
            )
            arch = inv["architecture_id"]
            require(
                arch not in signatures or signatures[arch] == identity,
                "Ambiguous architecture maps to incompatible method/task/context/parameters",
            )
            signatures[arch] = identity
            require(
                int(inv["window_size"]) == int(reg["window_points"]),
                "Cost inventory and authoritative registry contexts differ",
            )
            require(mapped["architecture_id"] == arch, "Cost mapping architecture identity differs")
            for field in ("protocol_id", "model_family", "variant", "seed"):
                if field in mapped:
                    require(
                        str(mapped[field]) == str(reg[field]),
                        "Cost mapping registry identity differs",
                    )
                if field in inv:
                    require(
                        str(inv[field]) == str(reg[field]),
                        "Cost inventory registry identity differs",
                    )
            representative = mapped["timing_representative_entry_id"]
            require(
                representative in inventory
                and int(registry[representative]["seed"]) == SEEDS[0]
                and inventory[representative]["architecture_id"] == arch,
                "Cost mapping representative checkpoint/architecture differs",
            )
            require(
                arch not in arch_representatives or arch_representatives[arch] == representative,
                "Ambiguous representative mapping for one architecture",
            )
            arch_representatives[arch] = representative
            if inv.get("timing_representative_entry_id"):
                require(
                    inv["timing_representative_entry_id"] == representative,
                    "Inventory and architecture mapping representatives differ",
                )
        timed = {}
        for record in source["report"]["records"]:
            require(
                record["device"] == settings["device"] and record["batch_size"] in batches,
                "Measured neural timing device/batch differs from frozen source protocol",
            )
            rep, arch = record["representative_entry_id"], record["architecture_id"]
            require(
                arch_representatives.get(arch) == rep,
                "Measured representative differs from authoritative entry mapping",
            )
            expected = inventory[rep]
            require(
                int(record["parameters"]) == int(expected["parameters"]),
                "Measured and inventoried parameter counts differ",
            )
            if "task" in record:
                require(
                    record["task"] == registry[rep]["task"], "Measured representative task differs"
                )
            if record.get("candidate_id"):
                require(
                    record["candidate_id"] == registry[rep]["model_family"],
                    "Measured representative method differs",
                )
            if "input_shape" in record:
                require(
                    record["input_shape"]
                    == [record["batch_size"], 6, int(expected["window_size"])],
                    "Measured representative window/input shape differs",
                )
            key = (arch, rep, record["device"], int(record["batch_size"]))
            require(key not in timed, "Ambiguous duplicate measured timing identity")
            timed[key] = record
        require(
            set(timed)
            == {
                (arch, rep, settings["device"], batch)
                for arch, rep in arch_representatives.items()
                for batch in batches
            },
            "Measured neural timing exact per-device batch coverage differs",
        )
        devices = {r["device"] for r in source["report"]["records"]}
        require(len(devices) == 1, "One latency inventory has ambiguous device identity")
        device = next(iter(devices))
        for jid, inv in inventory.items():
            reg = registry[jid]
            rep = mapping[jid]["timing_representative_entry_id"]
            context = temporal_row(
                {
                    **inv,
                    "job_id": jid,
                    "protocol": reg["protocol_id"],
                    "task": reg["task"],
                    "information_scope": "unified_six_low_neural",
                    "canonical_method": reg["model_family"],
                    "variant": reg["variant"],
                    "seed": reg["seed"],
                    "model_type": training_settings[jid].get("model_type", ""),
                    "window_points": int(inv["window_size"]),
                    "inventory_source": source_name,
                    "timing_representative_entry_id": rep,
                    "timing_representative_protocol": registry[rep]["protocol_id"],
                    "timing_representative_variant": registry[rep]["variant"],
                    "timing_reused_for_context": jid != rep,
                    "timing_reference_note": "One declared seed-20261010 representative for identical full architecture/input shape; mapping rows are not independent timing repeats.",
                }
            )
            if source_name == "latency_cpu":
                parameters.append(context)
            for batch in batches:
                key = (inv["architecture_id"], rep, device, batch)
                require(
                    key in timed, "Missing exact representative/device/batch timing for context"
                )
                record = timed[key]
                latency.append(
                    {
                        **context,
                        "device": device,
                        "precision": "float32",
                        "batch_size": batch,
                        "wall_p50_ms": record["wall"]["p50_ms"],
                        "wall_p95_ms": record["wall"]["p95_ms"],
                        "timing_calls": record["wall"]["count"],
                        "loop_throughput_windows_per_second": record[
                            "loop_throughput_windows_per_second"
                        ],
                        "timing_scope": "resident_normalized_FP32_network_forward",
                        "timing_source_member": source["source_member"],
                        "timing_source_sha256": source["source_sha256"],
                        "timing_protocol_json": json.dumps(settings, sort_keys=True),
                    }
                )
    return parameters, latency


def neural_cost_display_rows(rows):
    """Deduplicate only equal references across optimizer seeds, retaining contexts."""
    groups = defaultdict(list)
    for row in rows:
        groups[
            tuple(
                row[k]
                for k in (
                    "task",
                    "protocol",
                    "canonical_method",
                    "variant",
                    "window_points",
                    "architecture_id",
                    "device",
                    "batch_size",
                )
            )
        ].append(row)
    result = []
    for values in groups.values():
        reference = values[0]
        fields = (
            "parameters",
            "timing_representative_entry_id",
            "wall_p50_ms",
            "wall_p95_ms",
            "loop_throughput_windows_per_second",
            "temporal_class",
        )
        require(
            all(all(str(r[k]) == str(reference[k]) for k in fields) for r in values),
            "Different cost observations cannot be collapsed across optimizer seeds",
        )
        result.append(
            {
                **reference,
                "final_entry_ids_json": json.dumps(sorted(r["job_id"] for r in values)),
                "optimizer_models_referencing_timing": len(values),
            }
        )
    return result


def temporal_table_projection(rows, data):
    """Annotated counterpart to a source table; original source copies stay byte-exact."""
    projected = []
    jobs = data["pipeline"]["jobs"]
    for original in rows:
        row = dict(original)
        identity = next(
            (
                row[k]
                for k in ("job_id", "entry_id", "representative_entry_id", "id")
                if row.get(k) in jobs
            ),
            None,
        )
        if identity:
            job = jobs[identity]
            protocol, family = job["cell"].split("/")
            row.setdefault("protocol", protocol)
            row.setdefault("task", "navigation" if protocol.startswith("navigation") else "paired")
            # A classical timing may reference a neural job for its input context;
            # the explicit classical method still owns the temporal classification.
            explicit = row.get("canonical_method") or (
                row.get("method") if row.get("method") != "model" else None
            )
            if not explicit:
                explicit = row.get("model_family", row.get("family", family))
                require(explicit == family, "Temporal source method and job identity differ")
                row["canonical_method"] = family
            row["temporal_projection_job_id"] = identity
        elif not any(row.get(k) for k in ("canonical_method", "model_family", "family", "method")):
            return None  # Aggregate/raw observations link through the explicit context mapping.
        method = row.get("canonical_method") or row.get("method")
        if method == "model" or not method:
            method = row.get("model_family", row.get("family"))
        if method and "/" in method:
            method = method.split("/", 1)[0]
        row.setdefault("canonical_method", method)
        row.setdefault(
            "protocol", row.get("protocol_id", row.get("training_scope", row.get("direction", "")))
        )
        if "information_scope" not in row:
            row["information_scope"] = (
                "observed_high_reference"
                if method == "HIGH"
                else "common_identity"
                if method == "identity"
                else "unified_six_low_neural"
                if method in [*FAMILIES, "tartanimu_window_adapter"]
                else "navigation_baseline"
                if row.get("task") == "navigation" or str(row["protocol"]).startswith("navigation")
                else "source_domain_shared"
                if row.get("direction")
                else "known_device_bank"
            )
        if "variant" not in row:
            phase = jobs[identity]["phase"] if identity else ""
            row["variant"] = (
                phase.removeprefix("final_")
                if phase
                else "reference"
                if method in {"identity", "HIGH"}
                else "classical"
            )
        projected.append(temporal_row(row))
    return projected


def temporal_method_conditions(data):
    conditions = []
    for row in data["neural_parameters"]:
        conditions.append(
            {
                k: row[k]
                for k in (
                    "task",
                    "protocol",
                    "information_scope",
                    "canonical_method",
                    "variant",
                    "window_points",
                    "model_type",
                    "temporal_class",
                    "temporal_label",
                    "temporal_label_zh",
                    "temporal_note",
                    "temporal_order",
                )
            }
        )
    for original in data["tables"]["metrics_seed_summary"]:
        if original["method"] == "model":
            continue
        task = original["task"]
        row = {
            "task": task,
            "protocol": original["protocol_id"],
            "canonical_method": original["method"],
            "information_scope": "navigation_baseline"
            if task == "navigation"
            else "common_identity"
            if original["method"] == "identity"
            else "known_device_bank",
            "variant": "reference" if original["method"] == "identity" else "bank",
            "window_points": 80 if task == "navigation" else "current_or_same_stream_history",
            "model_type": "",
        }
        conditions.append(temporal_row(row))
    if data.get("shared"):
        for original in data["shared"]["tables"]["control_registry"]:
            if original["canonical_method"] == "identity":
                continue
            conditions.append(
                temporal_row(
                    {
                        **{k: original[k] for k in METHOD_KEY},
                        "task": "paired",
                        "window_points": "current_or_same_stream_history",
                        "model_type": "",
                    }
                )
            )
    for protocol in DOWNSTREAM_VIEWS:
        conditions.append(
            temporal_row(
                {
                    "task": "paired",
                    "protocol": protocol,
                    "information_scope": "observed_high_reference",
                    "canonical_method": "HIGH",
                    "variant": "reference",
                    "window_points": "observed_reference",
                    "model_type": "",
                }
            )
        )
    unique = {}
    fields = ("task", *METHOD_KEY, "window_points")
    for row in conditions:
        key = tuple(row[k] for k in fields)
        require(key not in unique or unique[key] == row, "Temporal condition identity collision")
        unique[key] = row
    return sorted(
        unique.values(),
        key=lambda r: (
            r["task"],
            r["protocol"],
            r["temporal_order"],
            r["information_scope"],
            r["canonical_method"],
            r["variant"],
            str(r["window_points"]),
        ),
    )


def _markdown_table(rows, fields=None):
    if not rows:
        return "无符合本节条件的记录。\n"
    fields = fields or list(rows[0])

    def value(item):
        if item is None:
            return ""
        if isinstance(item, float):
            return f"{item:.6g}"
        return str(item).replace("|", "\\|").replace("\n", "<br>")

    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    lines.extend(
        "| " + " | ".join(value(row.get(key, "")) for key in fields) + " |" for row in rows
    )
    return "\n".join(lines) + "\n"


def _primary_comparison(rows):
    """One row per dataset/unit/family; do not pool scopes or add unlike units."""
    lookup = {}
    for r in rows:
        if (
            r["split"] == "test"
            and r["aggregation_level"] == "dataset"
            and r["method"] == "model"
            and r["metric"] in PRIMARY
        ):
            key = tuple(
                r[k] for k in ("protocol_id", "dataset_id", "model_family", "metric", "unit")
            )
            require((key, r["variant"]) not in lookup, "Duplicate model comparison dataset row")
            lookup[(key, r["variant"])] = r
    result = []
    for key in sorted({key for key, _ in lookup}):
        default, tuned = (lookup[(key, v)] for v in VARIANTS)
        d, t = finite(default["mean"], "default"), finite(tuned["mean"], "tuned")
        result.append(
            {
                "protocol": key[0],
                "test_dataset": key[1],
                "family": key[2],
                "metric": key[3],
                "unit": key[4],
                "default_mean": d,
                "default_sample_sd": default["sample_sd"],
                "tuned_mean": t,
                "tuned_sample_sd": tuned["sample_sd"],
                "tuned_minus_default": t - d,
                "improvement_pct": 100 * (1 - t / d) if d > 1e-12 else "",
                "n_each": 3,
            }
        )
    return result


def _budget_rows(registry):
    groups = defaultdict(list)
    for row in registry:
        groups[(row["protocol_id"], row["model_family"], row["variant"])].append(row)
    return [
        {
            "protocol": p,
            "family": f,
            "variant": v,
            "runs": len(rows),
            "best_epochs": ",".join(r["best_epoch"] for r in sorted(rows, key=lambda r: r["seed"])),
            "actual_epochs": ",".join(
                r["actual_epochs"] for r in sorted(rows, key=lambda r: r["seed"])
            ),
            "early_stopped": sum(str(r["early_stopped"]).lower() == "true" for r in rows),
            "elapsed_total_s": sum(finite(r["elapsed_seconds"], "elapsed") for r in rows),
        }
        for (p, f, v), rows in sorted(groups.items())
    ]


def _section(title, paragraphs=(), rows=None, fields=None, figure=None):
    return {
        "title": title,
        "paragraphs": list(paragraphs),
        "rows": rows,
        "fields": fields,
        "figure": figure,
    }


def _shared_sections(data, links):
    shared = data["shared"]["tables"]
    _, rankings = scoped_signal_tables(data)
    sections = [
        _section(
            "共同信息主比较：统一神经与 protocol-shared classical",
            [
                "新增六个 paired 协议，每协议九个逻辑控制（54）；其中48个 learned-state slots、6个共同 identity 引用。每个方法在一个协议内只有一套跨器件状态，预测只使用六个 LOW 与同流 LOW 历史；不以 dataset/LOW/HIGH 器件身份路由系数。统一神经也仅用六 LOW，无器件 ID。主排名使用此共同信息条件，原 known-device bank 保留为补充组，不能把同名方法跨 scope 平均。",
                "本次评估逐字节导入此前冻结拟合的76个文件，对应54个逻辑控制、48个学习状态和6个 identity 引用；76个文件不是76个模型。原合同、选择集、TRAIN/VAL 支持、系数、逐文件 SHA、命令和 UTC 均保留，拟合只计一次；导入不构成新的拟合、重新选择或优化器续训，也不把原拟合耗时重复计入本次评估。",
                "原 bank 按 (dataset_id, low_sensor, high_sensor) 分别拟合/选择，并用已知身份选择状态，是额外的信息条件与多状态部署。六个协议各自使用自己的 TRAIN/VAL，不把不同 Penn 外层角色的 VAL 合并；原神经93选择/558最终、120迁移和原5下游视图全部保持。",
                "历史 bank 合同保留 dataset、split、strict 三份清单；共享合同另含原 HPO 已冻结的 assignment 清单。兼容核验分别绑定旧 bank 的三项路径/SHA与共享的四项支持证据，不向旧合同补造第四个字段。原 bank 证明仍对同一状态、选择和 TRAIN/VAL 帧检查物理行值、数量、顺序、搜索网格及状态身份，独立 assignment 证据不替代这些检查。",
                "基础共享源码与本次外部验证层分别记录身份：基础合同的源码指纹不能代表全部实际执行代码。来源索引同时保留验证层源码、独立执行 profile、拟合导入记录和实际 argv、环境、进程出生及退出收据；新增验证只处理元数据与清单兼容，downstream 和 CPU 计时继续使用原阶段命令。",
                "同 TRAIN 物理拆分不等于相同逐行监督曝光或损失：经典直接拟合每连续段全部 N 行；神经统计可读 TRAIN N 行，但梯度来自完整窗口。VAL/TEST 使用 C=floor(N/40)*40 前缀，段长不足320跳过并留记录。80/160/320窗口共享 C 支持；经典与神经的窗口重复权重、未来上下文、容量、训练目标并不相同。",
                "共享 OLS/ridge/FIR/ARX-style LOW lag/delta 使用 raw SI NumPy float64、normalization.mode=none_raw_si、fitted normalizer=0；不能与神经 TRAIN 标准化残差的FP32前向当作完全相同计算。ARX 历史名只代表 LOW lags + 当前 LOW 差分，不使用过去 HIGH 或预测 HIGH 的反馈。",
                "共享选择分数与神经 grouped score 的物理聚合一致：同 pair 的段先合并平方误差和行数再开方，各 gyro/accel RMSE 除同支持 identity（分母 max(B,1e-12)），再 pair、dataset 宏平均。经典 TRAIN 未加权最小二乘仍不是 dataset-balanced training loss。所有候选、严格更小才更新的 tie 顺序、near-zero和失败证据保留。",
                "所有经典方法 n=1、SD空；54/108/360是逻辑覆盖，不是重复训练。三神经种子分别训练，均值不是三模型 ensemble；调用时间分布SD不代表训练种子SD。六协议全 selection-set SHA 冻结后才读取新 TEST，历史测试已看过，定位仍是 retrospective common-task controls。",
                f"完整共享控制/源行/状态：{links['shared_control_registry']}；全部候选：{links['shared_candidate_history']}；方法与信息范围：{links['shared_method_scope']}；108 slots：{links['shared_signal_registry']}；360 jobs/720 split refs：{links['shared_neural_support_registry']}。",
                f"逐方法范围的全部指标：{links['scoped_signal_metrics']}；主/补充排名：{links['scoped_signal_rankings']}。",
            ],
        )
    ]
    grouped = defaultdict(list)
    for row in rankings:
        if (
            row["ranking_condition"] == "main_common_information"
            and row["protocol"] in PROTOCOLS[:6]
        ):
            grouped[
                (
                    row["protocol"],
                    row.get("dataset_id", ""),
                    row["metric"],
                    row["unit"],
                    row["temporal_order"],
                    row["temporal_label_zh"],
                )
            ].append(row)
    for (protocol, dataset, metric, unit, _order, temporal_label), rows in sorted(grouped.items()):
        display = [
            {
                "方法/变体": r["canonical_method"] + "/" + r["variant"],
                "信息范围": "NN 六LOW"
                if r["information_scope"] == "unified_six_low_neural"
                else "共同 identity"
                if r["information_scope"] == "common_identity"
                else "共享经典",
                "均值": f"{float(r['mean']):.6g}",
                "n / sample SD": f"{r['n']} / "
                + (
                    f"{float(r['sample_sd']):.4g}"
                    if r.get("sample_sd") not in (None, "")
                    else "不适用"
                ),
                "排名": r["rank"],
            }
            for r in rows
        ]
        sections.append(
            _section(
                f"共同信息排名：{protocol} / {dataset} / {metric} ({unit}) / {temporal_label}",
                [
                    "按时间信息条件分组展示；排名仍为同协议/测试域/指标的原整体描述性排名，不是组内重排或显著性检验。全部方法保留。"
                ],
                display,
            )
        )
    aliases = shared["alias_audit"]
    status_counts = {
        status: sum(r["alias_status"] == status for r in aliases)
        for status in sorted({r["alias_status"] for r in aliases})
    }
    sections.append(
        _section(
            "Penn 单 pair 别名与原 bank 补充条件",
            [
                "四个 Penn-only 协议必须由实际 TRAIN/VAL 确认为单 pair，不能仅凭协议名假定等价。exact_alias 需要同协议、原行/顺序、history、网格/tie与solver、归一化、超参数、数值系数内容以及共同支持预测的实际证明；整个来源 JSON SHA 可不同。",
                "numerically_equivalent 仅指冻结 rtol=1e-10、atol=1e-12 内一致，保留两份状态与差异；合法的不同最优参数保留为 not_alias，不强制同值。不能跨 Penn fold 借状态，也不能将 pooled 全TRAIN共享状态替换成其中 Penn 专用 bank。identity 共用一次，别名输出不能变成独立种子或重复证据。",
                "实际审计状态计数："
                + "; ".join(f"{key}={value}" for key, value in status_counts.items())
                + "。",
                f"逐项 alias proof、数值差异与来源：{links['shared_alias_audit']}；原 bank 与共享分组的完整指标/排名：{links['scoped_signal_rankings']}。原 aux2 原表与负面结果原样保留。",
            ],
        )
    )
    totals = {
        (r["protocol"], r["canonical_method"], r["information_scope"]): r
        for r in shared["bank_parameter_totals"]
    }
    parameter_rows = []
    for row in shared["control_registry"]:
        bank = totals.get((row["protocol"], row["canonical_method"], "known_device_bank"))
        require(bank is not None, "Actual whole-bank parameter total missing")
        parameter_rows.append(
            {
                "协议": row["protocol"],
                "方法": row["canonical_method"],
                "共享系数/bytes": f"{row['coefficient_count']} / {row['coefficient_bytes']}",
                "bank pairs": bank["pair_count"],
                "bank系数/bytes": f"{bank['bank_coefficient_count']} / {bank['bank_coefficient_bytes']}",
            }
        )
    sections.append(
        _section(
            "实际状态、整 bank 部署量与实际工作负载时间",
            [
                "以下为实际 stored learned coefficient scalars（含截距）与float64 bytes，不是统计自由度/有效DoF：bias=6，affine/ridge=42，FIR(T)=36T+6，ARX-style(T)=36(T+1)+6。LOW lag/delta 可线性冗余。每状态系数、normalizer、非拟合超参数、序列化artifact大小和exact numeric去重均单列；不把JSON大小当参数数目。",
                "bank总量按每协议/方法所有原 pair 的唯一有效状态求和，每pair只计一次，family/variant/seed/context重复引用不乘参数；精确数值alias的物理存储去重另列，不替代pair-sum部署总量。不同协议不一定同时部署，跨协议artifact总存储另列。",
                "实际CPU计时为同冻结TRAIN连续输入、同段预载history，seed=20261024、单BLAS线程、threadpoolctl=3.6.0核验，W=80/160/320，B=1/128，warmup30、100×3。计入lag/delta和算法前向；排除I/O、fit、选择、状态载入、history检索/初始化。保留原始时间、P50/P95、call distribution sample SD和独立loop吞吐。",
                "per-state是一个状态/一个pair输入；整 bank workload 是固定每pair各B窗、同顺序处理 P×B 输入，shared也重复调用同一系数处理完全相同输入。整循环实际测时，不能累加各pair P95，不能把P个pair总成本冒充单器件查询。P=1 exact numeric alias只计时一次并明确映射。",
                "神经是FP32、已归一化驻留输入的network-only；经典是raw-SI float64 feature+algorithm forward。精度和范围不同，参数量/时延不合成单一普遍优劣结论，不推断40Hz实时保证。",
                f"逐状态全部array/artifact量：{links['shared_state_inventory']}；部署总量/存储去重：{links['shared_bank_parameter_totals']}；per-state实际时间：{links['shared_latency_summary']}；整P×B实际时间：{links['shared_workload_latency']}；全部调用分布：{links['shared_latency_raw']}。",
            ],
            parameter_rows,
        )
    )
    downstream = scoped_downstream_rows(data)
    sections.append(
        _section(
            "共享经典的独立光学 GT 下游主比较",
            [
                "新增5视图×9共享方法×4跨度×SO(3)/strapdown两任务=360逻辑summary slots。前四视图是af/as/bf/bs外层录制，pooled/bs只作相同bs训练范围参考，不能当第五独立场景。主比较为20个神经family/variant（各3训练种子）+9共享经典+HIGH，共30方法/70流；known-device bank原组另列。",
                "沿用独立 AprilTag rig-body GT、原刚体标定、原source clock origin、相同GT有效支持/窗口和full supported segment keys。80/200/400/800点非重叠窗跨度1.975/4.975/9.975/19.975s。每窗/段仅用一次GT q/p初始化；v仅由开始前已存在的0.1s GT位置差估计，无过去支持同样跳过，不在传播中更新GT，不新拟合lag/旋转/bias。",
                "同冻结midpoint SO(3)、strapdown与g=[0,0,-9.80665]。保留endpoint/all-sample姿态、heading/tilt可用性、3D/水平位置、逐窗/段、不支持与全部退化。经典n=1/SD空，原文件历史0 SD不回写；HIGH不是下游上界，信号RMSE改善不保证独立GT收益。",
                f"主/补充范围合表及负改进：{links['scoped_downstream_comparison']}；新增全部360行：{links['shared_downstream_summary']}；逐窗：{links['shared_downstream_windows']}；连续段：{links['shared_downstream_segments']}。",
            ],
        )
    )
    for protocol in DOWNSTREAM_VIEWS:
        for task, unit in (("attitude", "deg"), ("strapdown", "m")):
            eligible = [
                r
                for r in downstream
                if r["protocol"] == protocol
                and r["task"] == task
                and r["information_scope"] != "known_device_bank"
            ]
            grouped = defaultdict(dict)
            for row in eligible:
                grouped[(row["information_scope"], row["canonical_method"], row["variant"])][
                    int(row["samples"])
                ] = row
            display = []
            for key, values in sorted(grouped.items()):
                display.append(
                    {
                        "方法": "["
                        + next(iter(values.values()))["temporal_label_zh"]
                        + "] "
                        + ("shared: " if key[0] == SHARED_SCOPE else "")
                        + key[1]
                        + "/"
                        + key[2],
                        **{
                            f"{h:g}s": "不支持"
                            if values[s]["mean"] == ""
                            else f"{values[s]['mean']:.5g}"
                            + (
                                f" ({values[s]['improvement_over_low_pct']:+.1f}%)"
                                if values[s]["improvement_over_low_pct"] != ""
                                else " (NA)"
                            )
                            for s, h in zip(HORIZON_SAMPLES, HORIZONS, strict=True)
                        },
                    }
                )
            sections.append(
                _section(
                    f"共享主比较：{protocol} / {task} endpoint RMSE ({unit})",
                    [
                        "括号为相对共同LOW改进，负值为退化；全部保留。共享/参考n=1，神经n=3，sample SD见完整合表。"
                    ],
                    display,
                )
            )
    return sections


def _sections(data, config, table_links, source_links, figures_root, out):
    synthetic = data["synthetic"]
    comparison = temporal_rows(_primary_comparison(data["tables"]["metrics_seed_summary"]))
    downstream = downstream_comparison(data["downstream"])
    counts = defaultdict(lambda: [0, 0, 0])
    for row in comparison:
        slot = counts[(row["metric"], row["unit"])]
        delta = row["tuned_minus_default"]
        slot[0 if delta < 0 else 1 if delta > 0 else 2] += 1
    evidence = [
        f"{metric}（{unit}）：调参均值低于默认 {n[0]} 项，高于默认 {n[1]} 项，相等 {n[2]} 项。各项按训练协议、测试数据域和模型家族分别比较；这些计数不是统计显著性检验。"
        for (metric, unit), n in sorted(counts.items())
    ]
    sections = [
        _section(
            "阅读范围与结果身份",
            [
                "本报告由已完成的不可变结果包生成，不运行模型或重新选择候选。共 93 个协议/模型单元、558 个最终训练槽位（默认/调参各 3 个训练随机种子）、120 个跨域神经模型评估、5 个下游视图。",
                "合成验证报告：所有数值仅为测试夹具，不能作为任何方法的实验成绩或科学结论。"
                if synthetic
                else "以下数值来自本次完成结果包；全部失败、退化和不支持项保留，不以历史结果补空。",
                f"核心 git：{data['pipeline'].get('git_commit', '未记录')}；核心 pipeline SHA256：{data['core_sha256']}；package manifest SHA256：{data['pkg'].manifest_sha256}。",
                "任务定位为 retrospective common-task adapters。先前已查看部分历史测试，因此不能称为全球预注册、全球 SOTA 或原论文完整复现。",
            ],
        )
    ]
    if data.get("shared"):
        sections[0]["paragraphs"].append(
            "本 profile 的 paired 主比较为统一六LOW神经与每协议单状态共享经典（54 logical/48 learned/6 identity、108信号slots、360新增下游slots、360 paired jobs/720支持引用）。原按器件身份路由的 known-device bank 单列补充；所有表按 information_scope/canonical_method/variant 区分，经典n=1。"
        )
    sections.append(
        _section(
            "逐方法的时间信息条件与比较单位",
            [
                "配对逐点输出：TCN、单向GRU、DeepGyro adapter（eval冻结BatchNorm）及全部点式/LOW过去lag和后向差分经典方法标为 Causal/逐点因果；PlainCNN、CNN-BiLSTM、Transformer、Informer、PatchTST、iTransformer、TLIO adapter 标为 Window/窗口未来上下文。",
                "分类仅针对已准备40Hz输入、eval模式与冻结归一化/BatchNorm统计量，不证明数据预处理、窗口调度、流式状态维护或端到端系统已经在线实时化。时间条件与 shared/known-device bank 的器件信息条件是正交字段。HIGH为单独的观测参考，不属于学习方法因果/窗口组。",
                "SmartPNT导航统一标为 Window endpoint/窗口终点位移预测；该标签描述给定窗口终点输出，不给其内部逐点特征套用配对任务的非因果类别。所有方法、数值、单位与原整体描述性排名保留；分组不产生新的组内替代排名。",
                f"完整逐方法/协议/上下文条件：{table_links['temporal_method_conditions']}；按558真实entry_id映射的参数：{table_links['neural_parameter_contexts']}；每entry与实际测速代表/device/batch关联：{table_links['neural_latency_contexts']}。",
                f"分类器源码：{source_links['temporal_helper']}；SHA256：{sha256_file(TEMPORAL_HELPER)}；完整工具/表索引：{table_links['tool_sources']} / {table_links['table_index']}。",
            ],
        )
    )
    sections.append(
        _section(
            "1. 输入、输出与两类任务",
            [
                "主任务：40 Hz 窗口的 6 个 LOW 通道（gx, gy, gz, ax, ay, az）映射为同步 HIGH 6 通道；陀螺单位 rad/s，加速度 m/s²。当前复核模型不输入温度、静态分支、设备身份 one-hot 或额外 GT。HIGH 是测得的器件参考信号，存在自身偏置，并非绝对无误差真值。",
                "实际学习目标为 y = norm_target(HIGH − LOW) 的标准化残差；推理为 hatHIGH = LOW + inverse_norm_target(model(norm_LOW(LOW)))。输入/残差的均值与标准差仅由 TRAIN 拟合，validation/test 复用固定统计，不在目标域重估。当前默认与调参均为新 6 通道协议，不把旧 9 通道模型冒充匹配默认。",
                "SmartPNT 是独立的导航回归任务：80 点（首尾跨度 1.975 s）六通道输入，输出固定局部 ENU 坐标中窗口首尾位移 3 向量（m）。它没有 HIGH 标签，不能把该任务分数当作 LOW→HIGH 升级的直接证明。",
                "每个家族采用统一任务适配器；DeepGyro、TLIO、TartanIMU 仅为当前共同输入/目标下的骨干适配，并非它们原论文传感器、滤波和导航系统的完整重建。",
            ],
        )
    )
    sections.append(
        _section(
            "1a. 骨干适配范围与结构限制",
            ["下表披露结构和任务头的差异，不把共同任务结果解读为原论文系统排名。"],
            [
                {
                    "家族": "DeepGyro adapter",
                    "本轮适配与限制": "保留 dilated backbone，RF=511 点，大于本轮80/160/320全部窗口；边界上下文限制须保留，不等于原始长时校正系统。",
                },
                {
                    "家族": "TLIO ResNet18 adapter",
                    "本轮适配与限制": "总 stride=32；80/160/320点分别约3/5/10个coarse特征后扩展到稠密输出。替换位移/不确定度双头；无原 heading处理、协方差训练和 stochastic-cloning滤波，含非因果卷积。",
                },
                {
                    "家族": "TartanIMU window adapter",
                    "本轮适配与限制": "仅SmartPNT导航80点；单窗口LSTM token，不复现原multihead、covariance、多窗口导航或预训练。",
                },
                {
                    "家族": "Informer adapter",
                    "本轮适配与限制": "ProbSparse encoder/distillation + 新dense任务头，非原完整forecasting encoder-decoder。本代码在构造时用固定sampling_seed生成采样索引，注册为buffer并随checkpoint保存；eval复用该索引，无逐forward随机采样。",
                },
                {
                    "家族": "PatchTST adapter",
                    "本轮适配与限制": "channel-independent patch encoder + channel mixer/稠密任务头，不是原预测任务完整复现。",
                },
                {
                    "家族": "iTransformer adapter",
                    "本轮适配与限制": "variate-token encoder + 重建头/channel mixer，无RevIN，非原完整forecasting。",
                },
                {
                    "家族": "TCN/CNN/GRU/CNN-BiLSTM/Transformer",
                    "本轮适配与限制": "统一任务下的通用网络实现，不声称是某篇论文完整复现；双向或全窗网络使用未来上下文，延迟另行披露。",
                },
            ],
        )
    )
    sections.append(
        _section(
            "2. 数据拆分、重复检查与时间对齐",
            [
                "ALiMU：同次记录的多个 LOW 共用 VN100，按物理录制分组。归档 dyn03、stat10 较短的输入前缀变体，保留 dyn04、stat11；不能把共用 HIGH 的多个设备对计为独立录制。基础训练 dyn01/02/04/05/06/stat11，验证 dyn07，测试 dyn08/09；是跨录制测试，并非未见过的设备型号测试。stat11 只在 TRAIN，没有独立静态测试；FusionPortable 未纳入本轮正式配对实验。",
                "PennCOSVIO 基础协议 af/as 训练、bf 验证、bs 测试；af/as/bf 另作外层留一录制轮转，bs 对应 penn_only。pooled 也在 bs 上评估，是相同录制的独立训练范围比较，不能作为第五个独立场景与四个 Penn 外层录制重复汇总。AL-only、Penn-only 与 pooled 各自重新训练，排名语义不同于 pooled 模型按测试域切片。",
                "SmartPNT-MSF：Traj2 与 Traj1 物理文件哈希重复，保留 Traj1/3/4 三个唯一轨迹；三种角色轮转用于检查训练/验证/测试敏感性，不构成三个独立环境。每个 run 的实际拆分、严格 manifest 与原始数据哈希均在来源索引中保留。",
                "统一到 40 Hz；较高采样率数据先使用 63 taps 因果 FIR 抗混叠，AL 原生 40 Hz 不额外滤波。必需信号、采样时间或 GT 支持中的缺口、非有限值及支持不连续处按冻结规则断段；物理采样时间和信号可用时间分开记录，不能用 FIR 可用延迟拟合测试时钟。源数据 manifest、坐标变换与处理配置决定具体有效范围。",
                "AL 处理表的 temperature_c=25.0 是导入器占位值，不能解释为实测温度；Penn 的该列为 NaN。temperature_c 与 match_dt_ms 仅作来源/对齐元数据，不用于温漂学习，其缺失不等于信号无效。两列必须与 canonical 支持的列存在性、空值掩码、非空值和行序完全一致，并拒绝 Inf；全部保存行的必需信号、时间、模型和基线输出仍须有限，VIRTUAL 与 model 六轴须精确一致，overlap 须为有限正整数。此次核验修复不删行、不填补信号，也不改变预测或窗口支持。",
            ],
        )
    )
    sections.append(
        _section(
            "3. 搜索、确认、早停与预算",
            [
                "每个单元固定 24 个候选；successive halving 在 8/24/80 epochs 预算下保留 24/8/2，提升阶段从头重训。搜索种子 20261024，入围的 2 个候选用 20261025 确认，按两个验证种子的均值选择。默认参数与调参参数均用 20261010/11/12 完整重训；测试分数不参与选择。",
                "最终预算上限 80 epochs，最优 epoch 由保存的验证准则确定，早停可能减少实际 epochs。不同家族窗口/结构/损失由各自冻结配置明确；训练曲线损失不跨不同权重相加或直接作性能排名。",
                "配对 checkpoint 准则为 dataset-balanced/device-pair grouped normalized gyro/accel RMSE 比值：各物理量先除以同支持上的 identity RMSE，再按既定分组均衡聚合；该无量纲选择分数不是把 rad/s 与 m/s² 直接相加。导航 checkpoint 使用物理 3D 窗口位移 RMSE（m）。配对验证使用共同320点支持，避免上下文不同改变验证样本集合。",
                "loss_profile 搜索 Huber-only、MSE-only 与 composite（导航限前两者）。配对默认 composite：Huber信号 gyro:acc=2:1，加 derivative=0.02、spectral=0.02、signal-integral proxy=0.005；非composite信号权重1:1，以上辅助权重为0，Allan权重始终0。signal-integral proxy 仅信号积分辅助项，不是SO(3)姿态或真实导航损失。",
                "冻结训练使用 AdamW；cosine_warmup，warmup=min(5,epoch_budget//4)，最终80epochs对应5epochs warmup，min_lr=1e-6，early-stopping patience=15，gradient clip norm=1。lr/weight_decay/batch/window等由各份冻结候选配置指定；这是本轮80epochs流程，未沿用旧60epochs/plateau规则。",
                f"报告逐份读取并校验558个冻结配置的训练标量，完整设置表：{table_links['frozen_training_settings']}。",
                f"完成 ledger：{data['summary'].get('fit_jobs')} 个 fit jobs，其中失败 {data['summary'].get('failed_fit_jobs')}；所有 search/confirmation/final 事件与错误类型、canonical 登记区间时长、最优 epoch 见完整附录。最终 canonical 登记区间时长合计 {sum(finite(r['elapsed_seconds'], 'elapsed') for r in data['registry']):.3f} s（日志区间；非实际 fit 合计或全搜索墙钟）。",
                "canonical elapsed_seconds 沿原 job.started_utc/completed_utc 计算；串行时包含 backend 与检查点核验，并行预取后首项可能包含显存探测和整组等待，后续项甚至在实际 fit 结束后才登记 started_utc。因此上游 CSV 原 elapsed_scope 标签不代表并行 worker 的真实训练区间，不据这些登记时长计算训练提速。独立性能会计附录使用 worker result 的 started_utc/completed_utc/wall_seconds（训练、验证与取证，非纯 GPU 计算），分别列出成功已消费、完成未消费、失败/清理、合成显存探测、评估门及两次原 partial 恢复成本；重叠 worker 时长不相加当端到端墙钟。",
                "表中 SD 使用 sample SD（ddof=1），描述不同训练随机种子所得结果的差异，包含初始化、训练shuffle/dropout等随机过程，不是狭义优化器内部波动，也不是录制级置信区间。固定Informer推理复用已保存索引，不做Monte Carlo采样；不据此声称跨硬件bitwise一致。确定性经典方法按物理行、完整输出及指标指纹去重，n=1，无虚构三种子 CI。",
                f"完整最终训练登记：{table_links['accuracy_run_registry']}；完整搜索/失败 ledger：{table_links['accuracy_all_fit_jobs']}；逐模型预算汇总：{table_links['training_budget']}。",
            ],
        )
    )
    sections.append(
        _section(
            "4. 默认与调参：全部模型、物理单位分开比较",
            [
                *evidence,
                "改进百分比以明确同单位基线为分母，分母接近零时留空；不把角速度、加速度与位置的不同量纲相加。主表按 dataset/pair/sequence 原定义计算；完整指标包括各轴、bias、MAE、RMSE 与相应支持数。",
                f"完整默认/调参比较：{table_links['primary_comparison']}；matched-seed 差值和 SD：{table_links['accuracy_matched_seed_delta_summary']}；原始 seed summaries：{table_links['accuracy_metrics_seed_summary']}。",
            ],
        )
    )
    if data.get("shared"):
        sections.extend(_shared_sections(data, table_links))
    # Every primary comparison is visible in the PDF, divided by one protocol/metric.
    groups = defaultdict(list)
    for row in comparison:
        groups[
            (
                row["protocol"],
                row["metric"],
                row["unit"],
                row["temporal_order"],
                row["temporal_label_zh"],
            )
        ].append(row)
    for (protocol, metric, unit, _order, temporal_label), rows in sorted(groups.items()):
        display = [
            {
                "测试域": r["test_dataset"],
                "模型": r["family"],
                "默认均值±SD": f"{r['default_mean']:.5g} ± {float(r['default_sample_sd']):.3g}",
                "调参均值±SD": f"{r['tuned_mean']:.5g} ± {float(r['tuned_sample_sd']):.3g}",
                "调参-默认": f"{r['tuned_minus_default']:.4g}",
            }
            for r in rows
        ]
        sections.append(
            _section(
                f"4a. {protocol} / {metric} ({unit}) / {temporal_label}",
                ["每项 n=3 训练随机种子；数值越低越好。所有家族均保留。"],
                display,
            )
        )
    sections.append(
        _section(
            "5. 不同训练范围排名与跨域泛化",
            [
                "model_rankings / rank_correlations 分别保留 pooled_training_evaluation_slices（相同 pooled 训练模型的测试域切片）、different_training_and_evaluation_datasets（不同训练与测试域），以及 same_target_different_training_scope（相同目标域、不同训练范围）。这三种语义不能互换为单一跨数据集排名。排名关联仅作描述，无跨录制独立性或显著性主张。",
                "数据集排名比较限定在相同任务、指标和评价口径内。ALiMU/Penn 的六轴 LOW→HIGH 配对与 SmartPNT 的固定 ENU 窗口位移回归具有不同任务目标；若将两类排名并列展示，差异同时涉及数据集、任务及其适配条件，不能归因于纯数据集效应，也不能作为同一 LOW→HIGH 任务的跨数据集泛化证据。",
                "零样本迁移包括 AL→Penn 与 Penn→AL，各 10 家族 × 默认/调参 × 3 种子，共 120。仅使用源域训练统计量/经典状态，不用目标测试标签校准或选择种子。target-trained reference 是目标域监督训练的比较参考，不属于 zero-shot 方法。",
                f"全部排名：{table_links['accuracy_model_rankings']}；排名相关：{table_links['accuracy_rank_correlations']}；全部跨域结果及目标训练参考：{table_links['transfer_seed_summary']}。",
            ],
        )
    )
    sections.append(
        _section(
            "6. 模型参数与 CPU/GPU 速度的公平解释",
            [
                "558 个最终 checkpoint 都有参数/缓冲区/输入形状清单。相同完整结构和输入窗口仅由固定种子 20261010 代表计时，结构到全部训练槽位的映射保留；没有依据测试性能或最快的一次挑代表。",
                "神经：FP32、eval/inference_mode、驻留且已归一化的合成输入；冻结CPU协议只测B=1，CUDA:0测B=1与128，CPU B=128未测。warmup=30、100 iterations×3 repeats；报告 P50/P95 与独立吞吐 loop。CPU 和 CUDA 墙钟（含同步）、CUDA event 分开；300 次计时分布不是 3 训练随机种子 SD。固定采样buffer构造、模型初始化、模型加载、磁盘 I/O、前处理和设备复制不在forward计时范围。",
                "网络参数 P 和注册缓冲区（registered buffers）不包括 checkpoint 中独立的外部 feature_stats/target_stats 元数据；相应的参数/缓冲区字节数也不等于完整 checkpoint 或全部部署状态的存储量。模型内部归一化算子仍在 model(x) 计时内，其注册参数和运行统计照常计入参数/缓冲区清单。这里的 dense forward 计时止于 [B,out,W] 输出，不含网络外特征/标签变换、导航任务的最后 head 选择和逆归一化，也不含配对任务的逆残差变换、LOW 相加及 overlap 融合；因此不能将其称为物理输出端点的端到端在线延迟。",
                "经典：CPU NumPy float64；固定源 TRAIN 连续段并预载同段所需历史，计入 lag/delta 特征与矩阵计算等规定算法范围，排除 I/O/状态拟合。FIR/ARX 的历史不能在每窗重置。其精度、计时边界与神经 FP32 network-only 不同；不据此宣称普遍硬件速度优劣、在线 40Hz 保证或把 CPU/GPU 数字混排。",
                f"神经 CPU 全表：{table_links['latency_cpu_latency_architectures']}；GPU：{table_links['latency_gpu_latency_architectures']}；全部 558 参数：{table_links['latency_cpu_checkpoint_inventory']}；经典原始和完整映射：{table_links['classical_classical_latency']}。",
            ],
        )
    )
    cost_groups = defaultdict(dict)
    for row in data["neural_cost_display"]:
        group = (
            row["task"],
            row["protocol"],
            row["device"],
            row["temporal_order"],
            row["temporal_label_zh"],
        )
        key = (
            row["canonical_method"],
            row["variant"],
            row["window_points"],
            row["architecture_id"],
        )
        cost_groups[group].setdefault(key, {})[row["batch_size"]] = row
    for (_task, protocol, device, _order, temporal_label), methods in sorted(cost_groups.items()):
        display = []
        for (method, variant, window, _arch), values in sorted(methods.items()):
            require(
                set(values) == ({1} if device == "cpu" else {1, 128}),
                "Body cost table must use exact frozen CPU/CUDA batch coverage",
            )
            display.append(
                {
                    "方法": f"{method}/{variant}; W={window}",
                    "参数": values[1]["parameters"],
                    "B=1 P50/P95 ms": f"{float(values[1]['wall_p50_ms']):.5g} / {float(values[1]['wall_p95_ms']):.5g}",
                    "B=128 P50/P95 ms": "未测（冻结CPU协议）"
                    if device == "cpu"
                    else f"{float(values[128]['wall_p50_ms']):.5g} / {float(values[128]['wall_p95_ms']):.5g}",
                }
            )
        sections.append(
            _section(
                f"6a. {protocol} / {device} / {temporal_label}：方法、上下文与参数/实测前向",
                [
                    "同一完整结构/输入形状可被多个协议或变体引用；仅固定代表实际测速，引用不是独立测量或多模型ensemble。此表按真实entry_id/声明代表/device/batch绑定，未以结构hash猜方法或混合上下文。",
                    f"全部参数与实测CPU B=1 / CUDA B=1、128上下文映射：{table_links['neural_parameter_contexts']} / {table_links['neural_latency_contexts']}；代表checkpoint、原始结构标识、逐种子引用、来源settings/SHA与scope均保留。CPU B=128仅显式标为未测，不生成虚构数值行。",
                ],
                display,
            )
        )
    sections.append(
        _section(
            "7. 原 known-device bank 补充组：独立光学 GT"
            if data.get("shared")
            else "7. 独立光学 GT：姿态与 6DoF 传播",
            [
                "Penn AprilTag 光学刚体位姿独立于 HIGH 陀螺参考；使用既有刚体外参映射到 ADIS/统一器件坐标。GT 原始时间 = unified timestamp + 源记录固定 origin；不拟合测试 lag、旋转、偏置或方法专属校准。[官方概述](https://daniilidis-group.github.io/penncosyvio/)报告光学定位约10cm，不赋予数值姿态精度。[官方格式](https://daniilidis-group.github.io/penncosyvio/file_format/)定义带秒时间戳、row-major的body(rig)-to-world 3×4位姿；坐标轴与刚体变换方向依据[官方外参](https://daniilidis-group.github.io/penncosyvio/extrinsic_calib/)。",
                "主指标：同一冻结 midpoint SO(3) 陀螺积分，与独立 GT 的测地角 endpoint RMSE（deg），并保留全窗误差。辅助：同一 6DoF strapdown，世界系加速度梯形积分，g=[0,0,-9.80665] m/s²，报告 3D/水平位置 endpoint RMSE（m）。",
                "所有方法共用预声明不重叠 80/200/400/800 点窗口，实际跨度 1.975/4.975/9.975/19.975 s；每窗 q/p 从 GT 初始化一次，初速度只由开始前 0.1 s 的 GT 位置差估计一次，无过去支持的起始窗跳过且计数。不在传播中更新 GT；连续支持段的 full-segment 结果也保留。40 Hz 缺口 >0.0375 s 断段，GT 插值最大间隔 0.1 s。",
                "每个外层录制保留 20 神经方法变体（各3种子）、9经典和 HIGH，共30方法、70信号流。LOW(identity)、HIGH 与所有虚拟输出算法、初始化、支持窗口完全一致。HIGH 只是带自身误差的器件参照，不保证是下游上界。",
                "同协议 Penn 光学 GT 下游组共 350 条信号流：300 条神经、45 条去重经典和 5 条 HIGH；5 个协议/GT 视图对应 4 条物理录制，不能视为 5 条独立录制。该组与 120 次跨域信号迁移评估是独立的实验范围；其模型来源覆盖 360 个配对最终模型中的 300 个 Penn 视图身份，不包括 AL-only 的 60 个模型在 Penn 上的跨域输出，也未将 120 次迁移输出用于光学 GT 下游重放。因此不能把同协议下游结果表述为已经验证的迁移下游收益。",
                "这属于 oracle 初始状态的离线诊断，不是完整导航系统、VIO 或可部署无 GT 初始化系统；不同窗口模型未来上下文与等待时延由元数据说明，不作因果实时能力推论。",
                f"全部方法、全部时长、全部退化：{table_links['downstream_comparison']}；window/segment/physical support 原始文件见来源索引。",
            ],
        )
    )
    for protocol in DOWNSTREAM_VIEWS:
        for task, unit in (("attitude", "deg"), ("strapdown", "m")):
            rows = [r for r in downstream if r["protocol"] == protocol and r["task"] == task]
            bymethod = defaultdict(dict)
            for row in rows:
                bymethod[row["method"]][row["samples"]] = row
            lowrows = [
                r
                for r in data["downstream"]
                if r["protocol"] == protocol and r["task"] == task and r["method"] == "identity"
            ]
            display = [
                {
                    "方法": "LOW/identity",
                    **{
                        f"{h:g}s": f"{float(next(r for r in lowrows if int(r['samples']) == s)['endpoint_rmse_deg_mean' if task == 'attitude' else 'position_endpoint_rmse_m_mean']):.5g}"
                        for s, h in zip(HORIZON_SAMPLES, HORIZONS, strict=True)
                    },
                }
            ]
            for method, values in sorted(bymethod.items()):
                display.append(
                    {
                        "方法": method,
                        **{
                            f"{h:g}s": f"{values[s]['mean']:.5g}"
                            + (
                                f" ({values[s]['improvement_over_low_pct']:+.1f}%)"
                                if values[s]["improvement_over_low_pct"] != ""
                                else " (NA)"
                            )
                            for s, h in zip(HORIZON_SAMPLES, HORIZONS, strict=True)
                        },
                    }
                )
            regressions = sum(r["behavior"] == "degraded" for r in rows)
            sections.append(
                _section(
                    f"7a. {protocol} / {task} endpoint RMSE ({unit})",
                    [
                        f"视图：{'同 bs 录制 pooled 参考，单列且不计入四外层场景' if protocol == 'pooled' else '独立外层留一录制'}。括号为相对 LOW 改进，负值即退化。本视图/任务 {regressions}/{len(rows)} 个方法×时长比较均值高于 LOW；保留全部。SD、窗口数和逐种子详情见完整表，不据窗口数量构造独立样本 CI。"
                    ],
                    display,
                )
            )
    sections.append(
        _section(
            "8. 限制、可观测性与可支持的结论",
            [
                "SmartPNT 原生 IMU 安装/物理轴定义尚不明确；加载器使用增量÷dt 推导角速率/加速度，位置与速度由 ECEF 转固定 ENU，但未把 IMU 三轴旋转到 ENU。不能把六通道直接宣称已知 ENU，也不能在无文档证据时断言精确车体轴。",
                "初速度不可观测：忽略地球自转的短时惯性模型中，p'(t)=p(t)+c t+d 与 p(t) 有同样加速度和旋转，因此理想 IMU 相同，但 Δp'=Δp+cT。若原生 IMU 属于随体坐标，绕重力轴的整体旋转还保留随体 IMU，而 ENU 位移发生旋转；无初始航向输入时存在 yaw 规约歧义。当前六通道输入不含初始速度/航向，跨轨迹变差可能来自这些不可唯一确定性及运动先验差异，不能只归因为模型结构不足。",
                "高重叠窗口的局部位移不得累加为完整轨迹 ATE/RPE；导航主指标为每窗 1.975 s 位移误差。三条唯一轨迹、四个 Penn 录制及共享 HIGH 多器件对限制外推范围；训练随机种子 SD 不度量场景抽样不确定性。",
                "纯 LOW→HIGH 信号校正的收益不保证传递为独立 GT 的姿态/位置收益。HIGH 自身偏置、低频漂移、加速度和重力耦合、GT 误差、有限的跨域覆盖均可能导致下游退化。本报告保留这些反证，不额外用目标测试 GT 校准来改善结果。",
                "能够主张的是：在固定公共数据、共同任务适配器与所列离线条件下的可复查比较；是否改善须逐任务、数据域、单位和时间尺度依据有限数值判断，不能扩大为全球 SOTA、普适器件升级或实际部署成功。",
            ],
        )
    )
    summary_figures = [
        f for f in data["figures"]["figures"] if f.get("coverage", {}).get("group") != "training"
    ]
    for fig in summary_figures:
        sections.append(
            _section(
                "图：" + fig["title"],
                [
                    fig["purpose"],
                    fig["caption"],
                    "来源、四格式导出与视觉 QA 见 figure_index.json。",
                ],
                figure=Path(figures_root) / fig["files"]["png"],
            )
        )
    sections.append(
        _section(
            "9. 完整附录与复现入口",
            [
                "全部比较表保留 CSV 与 Markdown 两份；正文的每项比较与附录来自同一已验证 source manifest，不读取 outputs/latest，不使用历史成绩补全。模型、预测 parquet、配置、训练/评估指标、data/split manifest、源码和独立 GT 均由 source_index 逐文件记录原始绝对身份、包内映射、SHA256、大小与链接。原 curated 132MB bulk 数据可由已校验原训练 bundle 恢复；报告不会把未导出的 bulk 文件伪称已在镜像中。",
                "历史 S1/S2 及 v1/v2 的失败记录、已产生的部分工件、once 令牌、控制台和已知成本均保留，不改写为成功。v1 的进程身份登记与清单兼容失败分别留证；v2 在初始化时因来源角色名的大写字符被原命名规则拒绝，v3 只规范角色标签，保留原文件名和内容。V3 正式评估前完成实际元数据初始化检验及只读准入：在独立的合成目标中验证41份 prior、76个冻结文件和8份源码的复制，再核验实际目标、来源与退休进程身份；这些预检不执行科学阶段，不替代 TEST、下游、计时或完整报告验收。失败、部分输出、导入与审计成本分别记录，未知成本保持未知，并行重叠时间不相加冒充墙钟时间。",
                "V3 的四个逻辑阶段均有完成记录，其中拟合选择沿用原已完成拟合，三个实际评估子进程均退出0；原父协调器随后在最终 identity 别名比较中因 KeyError('seed') 退出1。该故障来自 A1 记录的 seed/category/family/method/variant 与共享记录的 canonical_method/information_scope/variant/n/sd 等元数据外壳不同；原 identity 的 seed 明确为 null，不为其补造种子。独立的类型化校验覆盖层严格验证两套元数据角色、身份及完整字段集合，再按原容差逐字段比较全部原有科学与上下文字段，不改写记录或科学数值；原冻结协调器源码（SHA256：b28525bdc787d29e862bf4d23538c2db0d6981144b8f23426a679b5aef3180e4）保持逐字节不变。实际只读 dry-run 与随后按独立 profile、publication plan 执行的发布均重新通过原完整保存工件校验，覆盖20组、1136对 identity 记录（窗口记录1126对、片段记录10对）；独立发布的子进程和监督器均退出0，具有自身末尾成功标记及随后确认两者均已退出的退休证据。新发布状态与原失败状态分别留证，原父协调器退出1不改写为成功。随后 V8 导出准入因读取器额外要求历史 A1/A2 合同中不存在的 out 字段而拒绝，未启动导出；该失败与原源码均留证。独立 V9 读取器使用已封存的四/六字段完整合同、来源哈希和固定路径进行核验，后续真实只读完整准入已通过，科学工件保持不变。此过程不重新拟合、不重启信号评估、下游或 CPU 计时阶段，不生成新预测或重新测量时延；保存工件校验可重放固定经典核与指标。独立发布成功不替代后续实际导出工具的全量审计、导出与接收验收、完整报告及图件/PDF逐页视觉 QA，也不据此宣告全研究完成。",
                f"完整来源索引：{source_links['index']}；机器可读来源：{source_links['json']}；图索引及全部 93 单元训练曲线：{source_links['figures']}。",
                f"输入包manifest：{source_links['manifest']}；有限值/选择/来源完成审计：{source_links['audit']}；本报告配置：[config_snapshot](config_snapshot.yaml)，生成器：[build_reviewer_report.py](processing/build_reviewer_report.py)，包映射读取器：[reviewer_artifact_package.py](processing/reviewer_artifact_package.py)。输出完整哈希见[artifact_manifest](artifact_manifest.json)，输入/图/代码合同见[contract](contract.json)。",
                "推荐复现顺序：校验传输分片与 package manifest → 恢复源数据 bundle/校验 data manifest → 使用冻结环境与配置复现 pipeline → 验证 aux1/aux2 完成状态 → 生成全部图并实际视觉检查 → 执行本报告 CLI。报告不重新训练或推理。",
                "生成命令：python scripts/build_reviewer_report.py --results-root PACKAGE_MIRROR --figures-root FIGURES --out OUT；完成后 --resume 只校验并跳过，输入或输出改变会拒绝。",
            ]
            + [f"{name}：{link}" for name, link in sorted(table_links.items())],
        )
    )
    return sections, comparison, downstream


def _render_markdown(sections, title, out, synthetic):
    lines = [
        "# " + title,
        "",
        "**合成验证 / SYNTHETIC FIXTURE ONLY：不是实际实验成绩。**"
        if synthetic
        else "已完成冻结实验的可追溯复核报告。",
        "",
    ]
    for section in sections:
        lines.extend(["## " + section["title"], ""])
        for paragraph in section["paragraphs"]:
            lines.extend([paragraph, ""])
        if section["figure"]:
            lines.extend([f"![{section['title']}]({_relative(section['figure'], out)})", ""])
        if section["rows"]:
            lines.extend([_markdown_table(section["rows"], section["fields"]), ""])
    return "\n".join(lines)


def _render_html(sections, title, out, synthetic):
    chunks = [
        f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{html.escape(title)}</title><style>body{{font-family:"Microsoft YaHei",sans-serif;max-width:1180px;margin:30px auto;padding:0 24px;line-height:1.65;color:#17243b}}table{{border-collapse:collapse;width:100%;font-size:12px;margin:18px 0}}td,th{{border:1px solid #c9d1dc;padding:6px;overflow-wrap:anywhere}}th{{background:#e9f0f8}}img{{max-width:100%}}.notice{{background:#fff1d1;padding:16px}}a{{color:#12647b}}@media print{{thead{{display:table-header-group}}tr{{break-inside:avoid}}h2{{break-after:avoid}}}}</style><body><h1>{html.escape(title)}</h1>'
    ]
    if synthetic:
        chunks.append(
            '<p class="notice"><strong>合成验证 / SYNTHETIC FIXTURE ONLY：不是实际实验成绩。</strong></p>'
        )
    for section in sections:
        chunks.append("<h2>" + html.escape(section["title"]) + "</h2>")
        for paragraph in section["paragraphs"]:
            chunks.append("<p>" + _html_links(paragraph) + "</p>")
        if section["figure"]:
            chunks.append(
                f'<img src="{_relative(section["figure"], out)}" alt="{html.escape(section["title"])}">'
            )
        rows = section["rows"]
        if rows:
            fields = section["fields"] or list(rows[0])
            chunks.append(
                "<table><thead><tr>"
                + "".join("<th>" + html.escape(k) + "</th>" for k in fields)
                + "</tr></thead><tbody>"
            )
            chunks.extend(
                "<tr>"
                + "".join("<td>" + html.escape(str(row.get(k, ""))) + "</td>" for k in fields)
                + "</tr>"
                for row in rows
            )
            chunks.append("</tbody></table>")
    return "\n".join(chunks) + "</body></html>\n"


def _html_links(text):
    import re

    escaped = html.escape(text)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)


def _fit_figure_block(flowables, figure_image, canvas, *, frame_width, frame_height):
    """Fit the full title/caption/image group to one measured fresh PDF frame."""
    from reportlab.platypus import KeepTogether

    group = KeepTogether(flowables)
    group.wrapOn(canvas, frame_width, frame_height)
    before = float(group._H)
    original_height, original_width = figure_image.drawHeight, figure_image.drawWidth
    scale = 1.0
    if before > frame_height:
        # Paragraph wrapping/spacing uses the same font and inner-frame width as
        # rendering. Only the image scales; every caption character is retained.
        caption_and_spacing = before - original_height
        available = frame_height - caption_and_spacing - 6.0
        scale = available / original_height
        require(scale >= 0.75, "Full caption leaves insufficient room for a readable figure")
        figure_image.drawHeight = original_height * scale
        figure_image.drawWidth = original_width * scale
        group = KeepTogether(flowables)
        group.wrapOn(canvas, frame_width, frame_height)
    require(
        float(group._H) <= frame_height + 1e-7,
        "Measured complete figure/caption group still exceeds the PDF frame",
    )
    return group, {
        "frame_width_pt": frame_width,
        "frame_height_pt": frame_height,
        "original_block_height_pt": before,
        "final_block_height_pt": float(group._H),
        "image_scale_from_initial_display": scale,
        "displayed_image_width_pt": figure_image.drawWidth,
        "displayed_image_height_pt": figure_image.drawHeight,
        "caption_preserved": True,
        "minimum_allowed_image_scale": 0.75,
    }


def _pdf(sections, title, out, config, synthetic):
    from io import BytesIO

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import (
        Image,
        LongTable,
        NotAtTopPageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        TableStyle,
    )

    font = "ReportCJK"
    registered = False
    selected_font_path = None
    for candidate in config["font_candidates"]:
        if Path(candidate).is_file():
            try:
                selected_font = TTFont(font, candidate, subfontIndex=0)
                require(
                    all(ord(char) in selected_font.face.charToGlyph for char in "中文初速度"),
                    "Font lacks required Chinese glyphs",
                )
                pdfmetrics.registerFont(selected_font)
                registered = True
                selected_font_path = candidate
                break
            except Exception:
                continue
    if not registered:
        font = "STSong-Light"
        pdfmetrics.registerFont(UnicodeCIDFont(font))
    styles = {
        "body": ParagraphStyle(
            "body",
            fontName=font,
            fontSize=9,
            leading=15,
            wordWrap="CJK",
            spaceAfter=9,
            alignment=TA_LEFT,
        ),
        "heading": ParagraphStyle(
            "heading",
            fontName=font,
            fontSize=14,
            leading=21,
            spaceBefore=14,
            spaceAfter=10,
            wordWrap="CJK",
            keepWithNext=True,
        ),
        "title": ParagraphStyle(
            "title", fontName=font, fontSize=23, leading=32, spaceAfter=20, wordWrap="CJK"
        ),
        "cell": ParagraphStyle("cell", fontName=font, fontSize=7.4, leading=10.6, wordWrap="CJK"),
    }
    width = A4[0] - 84
    frame_width, frame_height = width - 12, A4[1] - 42 - 45 - 12
    measuring_canvas = Canvas(BytesIO(), pagesize=A4)
    figure_fit_qa = []
    story = [
        Paragraph(html.escape(title), styles["title"]),
        Paragraph(
            "合成验证 / SYNTHETIC FIXTURE ONLY；所有数值均为虚构夹具，不是研究结论。"
            if synthetic
            else "不可变来源包 · 完整覆盖 · 独立 GT 诊断",
            styles["body"],
        ),
    ]
    for section in sections:
        if section["title"].startswith(("1.", "3.", "5.", "6.", "7.", "8.", "9.")):
            # A trailing table spacer can spill onto a fresh frame and make it
            # appear nonempty to the chapter break, yielding a footer-only page.
            while story and isinstance(story[-1], Spacer):
                story.pop()
            story.append(NotAtTopPageBreak())
        section_start = len(story)
        story.append(Paragraph(html.escape(section["title"]), styles["heading"]))
        for paragraph in section["paragraphs"]:
            story.append(Paragraph(_html_links(paragraph), styles["body"]))
        if section["figure"]:
            from PIL import Image as PILImage

            with PILImage.open(section["figure"]) as image:
                w, h = image.size
            scale = min(frame_width / w, 440 / h)
            figure_image = Image(str(section["figure"]), width=w * scale, height=h * scale)
            story.append(figure_image)
            story.append(Spacer(1, 12))
        if section["rows"]:
            fields = section["fields"] or list(section["rows"][0])
            table_data = [[Paragraph(html.escape(str(k)), styles["cell"]) for k in fields]]
            table_data += [
                [Paragraph(html.escape(str(row.get(key, ""))), styles["cell"]) for key in fields]
                for row in section["rows"]
            ]
            if len(fields) == 5:
                fractions = (
                    [0.25, 0.1875, 0.1875, 0.1875, 0.1875]
                    if fields[0] == "方法"
                    else [0.14, 0.25, 0.22, 0.22, 0.17]
                )
            else:
                fractions = [1 / len(fields)] * len(fields)
            table = LongTable(
                table_data, colWidths=[width * n for n in fractions], repeatRows=1, hAlign="LEFT"
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6edf6")),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, colors.HexColor("#f5f7fa")],
                        ),
                        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c6ced9")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([table, Spacer(1, 12)])
        if section["figure"]:
            group, fit = _fit_figure_block(
                story[section_start:],
                figure_image,
                measuring_canvas,
                frame_width=frame_width,
                frame_height=frame_height,
            )
            story[section_start:] = [group]
            figure_fit_qa.append(
                {
                    "figure": str(section["figure"]),
                    "png_sha256": sha256_file(Path(section["figure"])),
                    **fit,
                }
            )
    doc = SimpleDocTemplate(
        str(out / "report.pdf"),
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=45,
        title=title,
        author="Reviewer artifact report builder",
    )

    def footer(canvas, document):
        canvas.setFont(font, 7)
        canvas.setFillColor(colors.HexColor("#516077"))
        canvas.drawString(
            42,
            23,
            "合成验证 SYNTHETIC ONLY" if synthetic else "冻结复核报告 · 详见来源索引与完整表",
        )
        canvas.drawRightString(A4[0] - 42, 23, str(document.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    from pypdf import PdfReader

    reader = PdfReader(out / "report.pdf")
    require(
        len(reader.pages) >= 5 and all(page.extract_text().strip() for page in reader.pages),
        "PDF pages/text incomplete",
    )
    footers = {"合成验证 SYNTHETIC ONLY", "冻结复核报告 · 详见来源索引与完整表"}
    require(
        all(
            any(
                line.strip() not in footers and not line.strip().isdigit() and line.strip()
                for line in page.extract_text().splitlines()
            )
            for page in reader.pages
        ),
        "PDF contains a footer-only blank page",
    )
    return {
        "pages": len(reader.pages),
        "font": font,
        "embedded_ttf": registered,
        "font_source": selected_font_path,
        "font_source_sha256": sha256_file(Path(selected_font_path)) if selected_font_path else None,
        "text_extraction_checked": True,
        "links": sum(len(p.get("/Annots", [])) for p in reader.pages),
        "figure_fit": figure_fit_qa,
        "visual_review_status": "pending",
        "render_status": "pending",
    }


def build_report(
    results_root, figures_root, out, *, config_path=None, resume=False, _allow_synthetic=False
):
    config_path = Path(config_path or DEFAULT_CONFIG).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    require(
        config.get("schema_version") == 1
        and config.get("require_visual_figure_review") is True
        and config.get("require_temporal_conditions") is True,
        "Report presentation schema/visual-review requirement differs",
    )
    data = audit_inputs(results_root, figures_root, _allow_synthetic=_allow_synthetic)
    out, figures_root = Path(out).resolve(), Path(figures_root).resolve()
    pkg = data["pkg"]
    require(
        not out.is_relative_to(pkg.root) and not out.is_relative_to(figures_root),
        "Report output must not modify immutable source package/figures",
    )
    contract = {
        "schema_version": 1,
        "analysis_profile": data["analysis_profile"],
        "package_manifest_sha256": pkg.manifest_sha256,
        "core_pipeline_sha256": data["core_sha256"],
        "package_files": pkg.files,
        "figure_files": data["figure_hashes"],
        "script_sha256": sha256_file(Path(__file__)),
        "temporal_support_sha256": sha256_file(TEMPORAL_HELPER),
        "package_reader_sha256": sha256_file(
            Path(__file__).with_name("reviewer_artifact_package.py")
        ),
        "config_sha256": sha256_file(config_path),
        "results_root": str(pkg.root),
        "figures_root": str(figures_root),
        "synthetic_fixture": data["synthetic"],
    }
    if out.exists():
        require(resume, "Report output exists; use explicit --resume")
        manifest = json.loads((out / "artifact_manifest.json").read_text(encoding="utf-8"))
        require(
            manifest["contract_sha256"] == canonical_hash(contract),
            "Report resume input/config/source contract changed",
        )
        for name, digest in manifest["files"].items():
            path = out / member_name(name)
            require(
                path.is_file() and sha256_file(path) == digest,
                "Report output checksum changed: " + name,
            )
        print("SKIPS: complete hash-verified report; no regeneration", flush=True)
        return json.loads((out / "summary.json").read_text(encoding="utf-8"))
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = out.parent / ("." + out.name + ".partial-" + uuid.uuid4().hex)
    staging.mkdir()
    json_write(
        staging / "PARTIAL.json",
        {
            "status": "building",
            "contract_sha256": canonical_hash(contract),
            "synthetic_fixture": data["synthetic"],
        },
    )
    try:
        (staging / "tables").mkdir()
        (staging / "processing").mkdir()
        for source in (
            Path(__file__),
            Path(__file__).with_name("reviewer_artifact_package.py"),
            TEMPORAL_HELPER,
        ):
            shutil.copyfile(source, staging / "processing" / source.name)
            expected = contract[
                "script_sha256"
                if source.name == Path(__file__).name
                else "temporal_support_sha256"
                if source == TEMPORAL_HELPER
                else "package_reader_sha256"
            ]
            require(
                sha256_file(staging / "processing" / source.name) == expected,
                "Report processing source changed during generation",
            )
        table_paths = {}
        temporal_derivatives = {}
        for prefix, directory in (
            ("accuracy", pkg.root_path("aux2") / "accuracy_tables"),
            ("transfer", pkg.root_path("aux1") / "transfer"),
            ("latency_cpu", pkg.root_path("aux1") / "latency_cpu"),
            ("latency_gpu", pkg.root_path("aux1") / "latency_gpu"),
            ("classical", pkg.root_path("aux2") / "classical_latency"),
        ):
            for member in sorted(pkg.files):
                path = pkg.resolve(member)
                if path.parent == directory and path.suffix == ".csv":
                    name = path.stem if prefix == "transfer" else prefix + "_" + path.stem
                    shutil.copyfile(path, staging / "tables" / (name + ".csv"))
                    require(
                        sha256_file(staging / "tables" / (name + ".csv")) == pkg.files[member],
                        "Copied comparison table changed during generation",
                    )
                    rows = _rows(path)
                    require(bool(rows), "Required complete table is empty: " + member)
                    (staging / "tables" / (name + ".md")).write_text(
                        "# "
                        + name
                        + "\n\n"
                        + ("合成夹具，不是实验成绩。\n\n" if data["synthetic"] else "")
                        + _markdown_table(rows),
                        encoding="utf-8",
                    )
                    table_paths[name] = "tables/" + name + ".csv"
                    projected = temporal_table_projection(rows, data)
                    if projected is not None:
                        temporal_derivatives["temporal_" + name] = projected
        comparison = _primary_comparison(data["tables"]["metrics_seed_summary"])
        derived = {
            "frozen_training_settings": data["training_settings"],
            "primary_comparison": temporal_rows(comparison),
            "training_budget": temporal_rows(_budget_rows(data["registry"])),
            "downstream_comparison": temporal_rows(
                downstream_comparison(data["downstream"]), default_task="paired"
            ),
            "downstream_all_summary": temporal_rows(data["downstream"], default_task="paired"),
            "neural_parameter_contexts": data["neural_parameters"],
            "neural_latency_contexts": data["neural_latency_contexts"],
            "neural_cost_display_contexts": data["neural_cost_display"],
            "temporal_method_conditions": temporal_method_conditions(data),
            "tool_sources": [
                {"source": name, "sha256": contract[key], "output_member": "processing/" + name}
                for name, key in (
                    (Path(__file__).name, "script_sha256"),
                    ("reviewer_artifact_package.py", "package_reader_sha256"),
                    (TEMPORAL_HELPER.name, "temporal_support_sha256"),
                )
            ],
        }
        if data["shared"]:
            for role, descriptor in data["shared"]["descriptors"].items():
                source = data["shared"]["root"] / member_name(descriptor["path"])
                name = "shared_" + role
                shutil.copyfile(source, staging / "tables" / (name + ".csv"))
                require(
                    sha256_file(staging / "tables" / (name + ".csv")) == descriptor["sha256"],
                    "Copied shared source table changed",
                )
                (staging / "tables" / (name + ".md")).write_text(
                    "# "
                    + name
                    + "\n\n"
                    + ("合成夹具，不是实验成绩。\n\n" if data["synthetic"] else "")
                    + _markdown_table(data["shared"]["tables"][role]),
                    encoding="utf-8",
                )
                table_paths[name] = "tables/" + name + ".csv"
                projected = temporal_table_projection(data["shared"]["tables"][role], data)
                if projected is not None:
                    temporal_derivatives["temporal_" + name] = projected
            metrics, rankings = scoped_signal_tables(data)
            derived.update(
                scoped_signal_metrics=metrics,
                scoped_signal_rankings=rankings,
                scoped_downstream_comparison=scoped_downstream_rows(data),
            )
        derived.update(temporal_derivatives)
        for name, rows in derived.items():
            write_csv(staging / "tables" / (name + ".csv"), rows)
            (staging / "tables" / (name + ".md")).write_text(
                "# "
                + name
                + "\n\n"
                + ("合成夹具，不是实验成绩。\n\n" if data["synthetic"] else "")
                + _markdown_table(rows),
                encoding="utf-8",
            )
            table_paths[name] = "tables/" + name + ".csv"
        table_index = [
            {
                "table": name,
                "csv": path,
                "markdown": path[:-4] + ".md",
                "rows": len(_rows(staging / path)),
                "sha256": sha256_file(staging / path),
                "temporal_annotations": "temporal_class" in _rows(staging / path)[0],
                "temporal_support_sha256": contract["temporal_support_sha256"],
                "raw_source_copy_preserved": name not in derived,
            }
            for name, path in sorted(table_paths.items())
        ]
        write_csv(staging / "tables/table_index.csv", table_index)
        (staging / "tables/table_index.md").write_text(
            "# Table index\n\n" + _markdown_table(table_index), encoding="utf-8"
        )
        table_paths["table_index"] = "tables/table_index.csv"
        source_index = [
            {
                "member": name,
                "source_identity": pkg.source_path(name),
                "sha256": digest,
                "size_bytes": pkg.manifest["file_metadata"][name]["size_bytes"],
                "local_path": str(pkg.resolve(name)),
                "link": _relative(pkg.resolve(name), out),
                "role": pkg.manifest["file_metadata"][name].get(
                    "role",
                    "generated artifact" if pkg.source_path(name) is None else "immutable source",
                ),
            }
            for name, digest in sorted(pkg.files.items())
        ]
        write_csv(staging / "source_index.csv", source_index)
        json_write(staging / "source_index.json", source_index)
        source_md = [
            {
                "包内路径": r["member"],
                "SHA256": r["sha256"],
                "原始身份": r["source_identity"] or "generated artifact",
                "文件": f"[查看]({r['link']})",
            }
            for r in source_index
        ]
        (staging / "source_index.md").write_text(
            "# 完整来源索引\n\n" + _markdown_table(source_md), encoding="utf-8"
        )
        links = {
            name: f"[CSV]({quote(path)}) / [Markdown]({quote(path[:-4] + '.md')})"
            for name, path in table_paths.items()
        }
        source_links = {
            "index": "[逐文件索引](source_index.md)",
            "json": "[JSON](source_index.json)",
            "figures": f"[figure_index]({_relative(figures_root / 'figure_index.json', out)})",
            "manifest": f"[package_manifest]({_relative(pkg.manifest_path, out)})",
            "audit": f"[completion_audit]({_relative(pkg.resolve('metadata/completion_audit.json'), out)})",
            "temporal_helper": "[reviewer_temporal_support.py](processing/reviewer_temporal_support.py)",
        }
        sections, _, _ = _sections(data, config, links, source_links, figures_root, out)
        title = config["title"] + ("（合成验证）" if data["synthetic"] else "")
        (staging / "report.md").write_text(
            _render_markdown(sections, title, out, data["synthetic"]), encoding="utf-8"
        )
        (staging / "report.html").write_text(
            _render_html(sections, title, out, data["synthetic"]), encoding="utf-8"
        )
        pdf_qa = _pdf(sections, title, staging, config, data["synthetic"])
        summary = {
            "schema_version": 1,
            "analysis_profile": data["analysis_profile"],
            "temporal_support_sha256": contract["temporal_support_sha256"],
            "status": "synthetic_fixture_complete" if data["synthetic"] else "complete",
            "synthetic_fixture": data["synthetic"],
            "created_utc": datetime.now(UTC).isoformat(),
            "selected_cells": 93,
            "final_runs": 558,
            "transfer_models": 120,
            "downstream_views": 5,
            "downstream_outer_recordings": 4,
            "downstream_method_variants_per_view": 30,
            "downstream_signal_streams_per_view": 70,
            "tables": len(table_paths),
            "source_files": len(source_index),
            "manifest_sha256": pkg.manifest_sha256,
            "contract_sha256": canonical_hash(contract),
            "pdf_pages": pdf_qa["pages"],
            "no_inference_or_fitting": True,
            "report_visual_review": "pending",
            "report_environment": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("reportlab", "pypdf", "Pillow")},
            },
        }
        if data["shared"]:
            summary.update(SHARED_COUNTS)
            summary.update(
                neural_split_support_references=720,
                shared_controls_index_sha256=data["shared"]["index_sha256"],
            )
        json_write(staging / "summary.json", summary)
        json_write(staging / "contract.json", contract)
        shutil.copyfile(config_path, staging / "config_snapshot.yaml")
        json_write(
            staging / "report_qa.json",
            {
                "status": "machine_checks_passed_visual_review_pending",
                "synthetic_fixture": data["synthetic"],
                "source_integrity": "passed",
                "coverage": {**ORIGINAL_COUNTS, **(SHARED_COUNTS if data["shared"] else {})},
                "figures_visual_review": "passed",
                "pdf": pdf_qa,
                "note": "Render and inspect report PDF after generation; this machine record does not assert a visual review.",
            },
        )
        (staging / "PARTIAL.json").unlink()
        files = {
            p.relative_to(staging).as_posix(): sha256_file(p)
            for p in staging.rglob("*")
            if p.is_file()
        }
        json_write(
            staging / "artifact_manifest.json",
            {
                "schema_version": 1,
                "status": summary["status"],
                "contract_sha256": canonical_hash(contract),
                "synthetic_fixture": data["synthetic"],
                "files": files,
            },
        )
        staging.rename(out)
        print(
            f"WRITES: {out} ({pdf_qa['pages']} PDF pages, {len(table_paths)} complete CSV/Markdown tables)",
            flush=True,
        )
        return summary
    except Exception as error:
        json_write(
            staging / "PARTIAL.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "contract_sha256": canonical_hash(contract),
                "synthetic_fixture": data["synthetic"],
            },
        )
        print(f"WRITES: failed partial report preserved at {staging}", flush=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--figures-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(
        f"READS: exact package {args.results_root}; figure index and QA {args.figures_root}; config {args.config}",
        flush=True,
    )
    print(
        f"WRITES: immutable Chinese Markdown/HTML/PDF, full tables and sources under {args.out}",
        flush=True,
    )
    print(
        "SKIPS: inference, training, fitting, test selection, historical-result fallback and synthetic relaxation",
        flush=True,
    )
    return build_report(
        args.results_root, args.figures_root, args.out, config_path=args.config, resume=args.resume
    )


if __name__ == "__main__":
    main()
