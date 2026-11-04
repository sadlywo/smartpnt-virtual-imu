"""Local display metadata/joins only; no real data/model/scientific execution."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from test_reviewer_report import report

TEMPORAL_FIELDS = {
    "temporal_class",
    "temporal_label",
    "temporal_label_zh",
    "temporal_note",
    "temporal_order",
}


@pytest.mark.parametrize(
    ("task", "method", "model_type", "expected"),
    [
        ("paired", "tcn", "tcn", "sample_causal_eval"),
        ("paired", "deepgyro_adapter", "deepgyro_backbone", "sample_causal_eval"),
        ("paired", "plain_cnn", "plain_cnn", "window_lookahead_eval"),
        ("paired", "ridge_tuned", None, "sample_causal_eval"),
        ("navigation", "plain_cnn", "plain_cnn", "window_endpoint"),
        ("navigation", "ridge_tuned", None, "window_endpoint"),
    ],
)
def test_temporal_projection_preserves_information_identity_and_numbers(
    task, method, model_type, expected
):
    original = {
        "task": task,
        "protocol": "navigation_rotation_0" if task == "navigation" else "pooled",
        "information_scope": "unified_six_low_neural" if model_type else "known_device_bank",
        "canonical_method": method,
        "variant": "tuned" if model_type else "bank",
        "model_type": model_type or "",
        "mean": -0.125,
        "rank": 7,
        "unit": "m" if task == "navigation" else "rad/s",
        "n": 3 if model_type else 1,
        "sample_sd": 0.2 if model_type else "",
    }
    result = report.temporal_row(original)
    assert result["temporal_class"] == expected
    assert {k: v for k, v in result.items() if k not in TEMPORAL_FIELDS} == original
    assert "not an end-to-end online guarantee" in result["temporal_note"]


def test_reference_and_downstream_stream_classification_are_explicit():
    high = report.temporal_row({"task": "strapdown", "method": "HIGH", "mean": 1.2, "n": 1})
    assert high["temporal_class"] == "measured_reference" and high["temporal_order"] == 3
    cnn = report.temporal_row({"task": "attitude", "method": "plain_cnn/tuned", "mean": 2.0})
    assert cnn["temporal_class"] == "window_lookahead_eval"
    with pytest.raises(ValueError, match="model_type"):
        report.temporal_row(
            {"task": "paired", "canonical_method": "gru", "model_type": "cnn_bilstm"}
        )
    with pytest.raises(ValueError, match="Unknown"):
        report.temporal_row({"task": "paired", "canonical_method": "unregistered"})
    with pytest.raises(ValueError, match="Unknown temporal source task"):
        report.temporal_row({"task": "unregistered_task", "canonical_method": "tcn"})
    with pytest.raises(ValueError, match="navigation protocol"):
        report.temporal_row(
            {"task": "paired", "protocol": "navigation_rotation_0", "canonical_method": "tcn"}
        )


def test_overall_ranks_units_scopes_and_identity_are_retained():
    base = {
        "task": "paired",
        "protocol_id": "pooled",
        "split": "test",
        "aggregation_level": "dataset",
        "dataset_id": "toy",
        "low_sensor": "__all_devices__",
        "high_sensor": "__all_devices__",
        "sequence_id": "__all_sequences__",
        "metric": "gyro_vector_rmse_radps",
        "unit": "rad/s",
        "n": 3,
        "sample_sd": 0.1,
        "support_count": 44880,
    }
    old = [
        {**base, "method": "model", "model_family": family, "variant": variant, "mean": value}
        for family, variant, value in (
            ("tcn", "default", 1.0),
            ("tcn", "tuned", 1.5),
            ("plain_cnn", "default", 2.0),
            ("plain_cnn", "tuned", 2.5),
        )
    ]
    old += [
        {
            **base,
            "method": method,
            "model_family": method,
            "variant": "classical",
            "mean": value,
            "n": 1,
            "sample_sd": "",
        }
        for method, value in (("identity", 3.0), ("ridge_tuned", 0.5))
    ]
    shared = [
        {
            "protocol": "pooled",
            "information_scope": report.SHARED_SCOPE,
            "canonical_method": method,
            "variant": "shared",
            "split": "test",
            "metric_scope": "dataset",
            "dataset_id": "toy",
            "low_sensor": "__all_devices__",
            "high_sensor": "__all_devices__",
            "sequence_id": "__all_sequences__",
            "metric": base["metric"],
            "value": value,
            "n": 1,
            "sd": "",
        }
        for method, value in (("identity", 3.0), ("ridge_tuned", 4.0))
    ]
    shared.append(
        {
            "protocol": "pooled",
            "information_scope": report.SHARED_SCOPE,
            "canonical_method": "identity",
            "variant": "shared",
            "split": "test",
            "metric_scope": "dataset",
            "dataset_id": "toy",
            "low_sensor": "__all_devices__",
            "high_sensor": "__all_devices__",
            "sequence_id": "__all_sequences__",
            "metric": "samples",
            "value": 44880,
            "n": 1,
            "sd": "",
        }
    )
    source = {
        "tables": {"metrics_seed_summary": old},
        "shared": {"tables": {"metrics_long": shared}},
    }
    untouched = copy.deepcopy(source)
    rows, ranked = report.scoped_signal_tables(source)
    assert source == untouched
    main = [r for r in ranked if r["ranking_condition"] == "main_common_information"]
    assert {(r["canonical_method"], r["variant"]): r["rank"] for r in main} == {
        ("tcn", "default"): 1,
        ("tcn", "tuned"): 2,
        ("plain_cnn", "default"): 3,
        ("plain_cnn", "tuned"): 4,
        ("identity", "reference"): 5,
        ("ridge_tuned", "shared"): 6,
    }
    assert all(r["unit"] == "rad/s" for r in rows if r["metric"] != "samples")
    assert len(
        [r for r in rows if r["canonical_method"] == "identity" and r["metric"] == base["metric"]]
    ) == 1
    assert {
        (r["information_scope"], float(r["mean"]))
        for r in rows
        if r["canonical_method"] == "ridge_tuned"
    } == {("known_device_bank", 0.5), (report.SHARED_SCOPE, 4.0)}
    assert all(r["sample_sd"] == "" for r in rows if r["n"] == 1)


def cost_fixture():
    registry = [
        {
            "job_id": jid,
            "protocol_id": protocol,
            "task": "paired",
            "model_family": "tcn",
            "variant": "default",
            "seed": report.SEEDS[0],
            "window_points": 80,
        }
        for jid, protocol in (("a", "pooled"), ("b", "alimu_only"))
    ]
    inventory = [
        {
            "entry_id": r["job_id"],
            "architecture_id": "same_arch",
            "parameters": 42,
            "window_size": 80,
            "timing_representative_entry_id": "a",
        }
        for r in registry
    ]
    mapping = [
        {
            "entry_id": r["job_id"],
            "architecture_id": "same_arch",
            "timing_representative_entry_id": "a",
        }
        for r in registry
    ]
    latency = {}
    for name, device, factor in (("latency_cpu", "cpu", 1), ("latency_gpu", "cuda:0", 2)):
        latency[name] = {
            "inventory": copy.deepcopy(inventory),
            "mapping": copy.deepcopy(mapping),
            "source_member": "synthetic/" + name + "/latency.json",
            "source_sha256": "a" * 64,
            "report": {
                "timing_protocol": {
                    "device": device,
                    "batch_sizes": [1] if device == "cpu" else [1, 128],
                    "cpu_threads": 1 if device == "cpu" else 4,
                    "input_seed": 20261024,
                    "end_to_end": False,
                    "warmup": 30,
                    "iterations": 100,
                    "repeats": 3,
                },
                "records": [
                    {
                        "architecture_id": "same_arch",
                        "representative_entry_id": "a",
                        "task": "paired",
                        "candidate_id": "tcn",
                        "parameters": 42,
                        "device": device,
                        "batch_size": batch,
                        "input_shape": [batch, 6, 80],
                        "wall": {
                            "p50_ms": factor * batch / 100,
                            "p95_ms": factor * batch / 50,
                            "count": 300,
                        },
                        "loop_throughput_windows_per_second": 1000 / factor,
                    }
                    for batch in ((1,) if device == "cpu" else (1, 128))
                ],
            },
        }
    return {
        "registry": registry,
        "training_settings": [{"job_id": r["job_id"], "model_type": "tcn"} for r in registry],
        "latency": latency,
    }


def test_classical_timing_context_does_not_inherit_neural_lookahead_label():
    source = [
        {
            "representative_entry_id": "cnn_job",
            "method": "arx_tuned",
            "variant": "bank",
            "information_scope": "known_device_bank",
            "wall_p95_ms": 0.125,
        }
    ]
    data = {
        "pipeline": {"jobs": {"cnn_job": {"cell": "pooled/plain_cnn", "phase": "final_default"}}}
    }
    result = report.temporal_table_projection(source, data)
    assert result[0]["temporal_class"] == "sample_causal_eval"
    assert all(result[0][k] == value for k, value in source[0].items())
    assert result[0]["canonical_method"] == "arx_tuned"


def test_enriched_source_table_has_explicit_method_scope_variant_without_changing_fields():
    original = {"entry_id": "j", "parameters": "42", "seed": "20261010"}
    result = report.temporal_table_projection(
        [original],
        {
            "pipeline": {
                "jobs": {"j": {"cell": "navigation_rotation_0/plain_cnn", "phase": "final_tuned"}}
            }
        },
    )[0]
    assert all(result[k] == value for k, value in original.items())
    assert (
        result["protocol"],
        result["information_scope"],
        result["canonical_method"],
        result["variant"],
    ) == ("navigation_rotation_0", "unified_six_low_neural", "plain_cnn", "tuned")
    assert result["temporal_class"] == "window_endpoint"


def test_cost_join_retains_reused_architecture_context_and_actual_values():
    data = cost_fixture()
    before = copy.deepcopy(data)
    parameters, timings = report.neural_cost_tables(data)
    assert data == before
    assert len(parameters) == 2 and len(timings) == 6
    assert {r["protocol"] for r in parameters} == {"pooled", "alimu_only"}
    assert all(r["canonical_method"] == "tcn" and r["window_points"] == 80 for r in timings)
    assert all(r["timing_representative_entry_id"] == "a" for r in timings)
    assert all(r["timing_reused_for_context"] for r in timings if r["job_id"] == "b")
    assert {(r["device"], r["batch_size"], r["wall_p50_ms"]) for r in timings} == {
        ("cpu", 1, 0.01),
        ("cuda:0", 1, 0.02),
        ("cuda:0", 128, 2.56),
    }
    assert len(report.neural_cost_display_rows(timings)) == 6
    assert all(r["timing_source_member"] and r["timing_source_sha256"] for r in timings)
    assert all(
        json.loads(r["timing_protocol_json"])["batch_sizes"] == [1]
        for r in timings
        if r["device"] == "cpu"
    )


@pytest.mark.parametrize(
    "mutation",
    ["missing_cpu1", "unexpected_cpu128", "missing_cuda128", "cpu_settings128", "record_device"],
)
def test_frozen_per_device_batch_and_source_settings_are_exact(mutation):
    data = cost_fixture()
    cpu, gpu = data["latency"]["latency_cpu"]["report"], data["latency"]["latency_gpu"]["report"]
    if mutation == "missing_cpu1":
        cpu["records"] = []
    elif mutation == "unexpected_cpu128":
        extra = copy.deepcopy(cpu["records"][0])
        extra.update(batch_size=128, input_shape=[128, 6, 80])
        cpu["records"].append(extra)
    elif mutation == "missing_cuda128":
        gpu["records"] = [r for r in gpu["records"] if r["batch_size"] == 1]
    elif mutation == "cpu_settings128":
        cpu["timing_protocol"]["batch_sizes"] = [1, 128]
    else:
        cpu["records"][0]["device"] = "cuda:0"
    with pytest.raises(ValueError, match=r"batch|device|coverage"):
        report.neural_cost_tables(data)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_entry",
        "different_representative",
        "family_context",
        "window",
        "timing_rep",
        "model_type",
    ],
)
def test_ambiguous_or_incompatible_cost_joins_fail_closed(mutation):
    data = cost_fixture()
    cpu = data["latency"]["latency_cpu"]
    if mutation == "duplicate_entry":
        cpu["mapping"].append(dict(cpu["mapping"][0]))
    elif mutation == "different_representative":
        cpu["mapping"][1]["timing_representative_entry_id"] = "b"
    elif mutation == "family_context":
        data["registry"][1]["model_family"] = "plain_cnn"
    elif mutation == "window":
        cpu["inventory"][1]["window_size"] = 160
    elif mutation == "timing_rep":
        cpu["report"]["records"][0]["representative_entry_id"] = "b"
    else:
        data["training_settings"][1]["model_type"] = "transformer"
    with pytest.raises(ValueError):
        report.neural_cost_tables(data)


@pytest.mark.parametrize("mutation", ["missing", "wrong", "snapshot"])
def test_old_or_wrong_figure_classifier_binding_is_rejected(tmp_path, mutation):
    manifest = "a" * 64
    index = {
        "schema_version": 1,
        "status": "complete",
        "synthetic": True,
        "synthetic_fixture": True,
        "manifest_sha256": manifest,
        "source_manifest_sha256": manifest,
        "qa_summary": {
            "status": "passed",
            "pdf_render_checked": True,
            "visual_review_status": "passed",
        },
        "coverage": {"training_cells": 93, "training_final_runs": 558},
        "source_files": {},
        "figures": [],
    }
    if mutation != "missing":
        helper_sha = report.sha256_file(report.TEMPORAL_HELPER)
        index["temporal_support_sha256"] = "b" * 64 if mutation == "wrong" else helper_sha
        index["temporal_support_source"] = {"path": "../escape.py", "sha256": helper_sha}
    (tmp_path / "figure_index.json").write_text(json.dumps(index), encoding="utf-8")
    pkg = SimpleNamespace(manifest_sha256=manifest)
    with pytest.raises(ValueError, match=r"temporal|Unsafe|path|Noncanonical"):
        report._figures_audit(pkg, tmp_path, True)


def test_nontraining_figure_qa_must_bind_current_temporal_classifier(tmp_path):
    helper_sha = report.sha256_file(report.TEMPORAL_HELPER)
    snapshot = tmp_path / "source_snapshot_reviewer_temporal_support.py"
    snapshot.write_bytes(report.TEMPORAL_HELPER.read_bytes())
    qa = tmp_path / "qa.json"
    qa.write_text(
        json.dumps({"status": "passed", "visual_review_status": "passed"}), encoding="utf-8"
    )
    files = {}
    for suffix in ("png", "pdf", "svg", "tiff"):
        p = tmp_path / ("fixture." + suffix)
        p.write_text("explicit synthetic rejection fixture", encoding="utf-8")
        files[suffix] = p.name
    manifest = "a" * 64
    index = {
        "schema_version": 1,
        "status": "complete",
        "synthetic": True,
        "synthetic_fixture": True,
        "manifest_sha256": manifest,
        "source_manifest_sha256": manifest,
        "qa_summary": {
            "status": "passed",
            "pdf_render_checked": True,
            "visual_review_status": "passed",
        },
        "coverage": {"training_cells": 93, "training_final_runs": 558},
        "source_files": {},
        "temporal_support_sha256": helper_sha,
        "temporal_support_source": {"path": snapshot.name, "sha256": helper_sha},
        "figures": [
            {
                "files": files,
                "sha256": {k: report.sha256_file(tmp_path / p) for k, p in files.items()},
                "qa": {
                    "status": "passed",
                    "visual_review_status": "passed",
                    "path": qa.name,
                    "sha256": report.sha256_file(qa),
                },
                "coverage": {"group": "metrics"},
            }
        ],
    }
    (tmp_path / "figure_index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="Nontraining figure QA"):
        report._figures_audit(SimpleNamespace(manifest_sha256=manifest), tmp_path, True)


# Actual 1,566-character caption that exposed a 43 pt overflow with the fixed
# 440 pt image cap. Keep this literal independent of ignored run artifacts.
LONG_TEMPORAL_CAPTION = (
    "SYNTHETIC VALIDATION ONLY. Main common-task comparison: unified six-channel neural models "
    "and one shared classical state per protocol, without device-ID coefficient routing. Neural "
    "n=3 optimization seeds; shared n=1, SD not applicable. Common LOW/identity appears once per "
    "comparison unit. TRAIN physical splits agree; raw-SI float64 OLS uses all N rows, whereas "
    "neural normalized FP32 window training has different loss/exposure. Context, causality and "
    "capacity can differ across algorithms. HIGH is a device proxy for signal panels; independent "
    "optical GT is used only for downstream propagation. ARX-style controls use LOW lag/delta, "
    "without HIGH feedback. All model variants and deduplicated classical methods; physical "
    "gyro/accel units remain separate. Neural: mean ± sample SD across 3 optimization seeds; "
    "classical: one deduplicated result; no recording CI. Method pages follow fixed family order; "
    "identical classical references repeat for comparison, never as extra statistical replicates. "
    "Temporal groups describe per-method inference context on prepared 40 Hz inputs in eval mode "
    "with frozen normalization/BatchNorm statistics. Causal uses current/past prepared samples; "
    "Window may use later samples within its supplied window. Navigation is window-endpoint "
    "displacement, not a paired per-sample causality claim. These labels are orthogonal to "
    "information scope and do not establish preprocessing/streaming or end-to-end online "
    "causality. Every method and original numerical value is retained; original overall ranks "
    "are not replaced by within-class ranks."
)


def test_actual_long_temporal_caption_and_tall_image_stay_together(tmp_path):
    from PIL import Image
    from pypdf import PdfReader

    png = tmp_path / "synthetic_tall.png"
    Image.new("RGB", (2160, 2790), "white").save(png)
    before = report.sha256_file(png)
    title = (
        "Protocol-shared main comparison · paired · penn_only · test · penncosyvio "
        "· Tango Top IMU · ADIS16488 · methods page1"
    )
    sections = [
        report._section(
            f"图: {title} ({i})",
            [
                "metrics",
                LONG_TEMPORAL_CAPTION,
                "合成时间条件/渲染 seam; 该图子集不提供93训练曲线的完整报告覆盖。",
            ],
            figure=png,
        )
        for i in range(5)
    ]
    config = json.loads(report.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    qa = report._pdf(sections, "合成时间条件长图注回归", tmp_path, config, True)
    pages = PdfReader(tmp_path / "report.pdf").pages
    assert len(pages) == 6
    assert not pages[0].images
    assert all(len(page.images) == 1 for page in pages[1:])
    for page in pages[1:]:
        text = "".join(page.extract_text().split())
        assert "".join(LONG_TEMPORAL_CAPTION.split()) in text
    assert all(fit["caption_preserved"] for fit in qa["figure_fit"])
    assert all(0.75 <= fit["image_scale_from_initial_display"] < 1 for fit in qa["figure_fit"])
    assert all(
        fit["final_block_height_pt"] <= fit["frame_height_pt"] - 5.9 for fit in qa["figure_fit"]
    )
    assert all(fit["displayed_image_width_pt"] <= fit["frame_width_pt"] for fit in qa["figure_fit"])
    assert all(
        fit["displayed_image_width_pt"] / fit["displayed_image_height_pt"]
        == pytest.approx(2160 / 2790)
        for fit in qa["figure_fit"]
    )
    assert report.sha256_file(png) == before


def test_unreadably_small_figure_is_rejected_instead_of_caption_truncation(tmp_path):
    from PIL import Image

    png = tmp_path / "synthetic_tall.png"
    Image.new("RGB", (2160, 2790), "white").save(png)
    section = report._section("图: 完整图注", [LONG_TEMPORAL_CAPTION * 3], figure=png)
    config = json.loads(report.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="insufficient room for a readable figure"):
        report._pdf([section], "合成超长图注应拒绝", tmp_path, config, True)
