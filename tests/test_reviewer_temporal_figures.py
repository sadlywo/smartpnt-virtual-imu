"""Fictional inference-context groups must not alter scientific quantities."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from test_reviewer_figures import module
from test_reviewer_shared_figures import PENN_IDENTITY
from test_reviewer_shared_figures import fixture as shared_fixture

FAMILY_TYPES = {
    "tcn": "tcn",
    "plain_cnn": "plain_cnn",
    "gru": "gru",
    "cnn_bilstm": "cnn_bilstm",
    "transformer": "transformer",
    "informer": "informer",
    "patchtst": "patchtst",
    "itransformer": "itransformer",
    "deepgyro_adapter": "deepgyro_backbone",
    "tlio_resnet18_adapter": "tlio_resnet18",
}


def fixture():
    """All paired families/variants, both scopes, CPU/GPU and endpoint semantics."""
    tool = module()
    tables, _, _ = shared_fixture()
    for name in ("metrics", "metrics_shared", "transfer"):
        frame = tables[name]
        prototype = frame[frame.method.eq("model") & frame.model_family.eq("transformer")]
        neural = [
            prototype.assign(model_family=family, model_type=model_type)
            for family, model_type in FAMILY_TYPES.items()
        ]
        tables[name] = pd.concat([*neural, frame[frame.method.ne("model")]], ignore_index=True)
    delta = tables["deltas"][tables["deltas"].model_family.eq("transformer")]
    tables["deltas"] = pd.concat(
        [
            delta.assign(model_family=family, model_type=model_type)
            for family, model_type in FAMILY_TYPES.items()
        ],
        ignore_index=True,
    )
    rank = tables["ranks"][tables["ranks"].model_family.eq("transformer")]
    tables["ranks"] = pd.concat(
        [
            rank.assign(model_family=family, rank=number)
            for number, family in enumerate(FAMILY_TYPES, 1)
        ],
        ignore_index=True,
    )
    latency = tables["latency_neural"]
    prototype = latency[latency.model_family.eq("transformer")]
    paired = pd.concat(
        [
            prototype.assign(model_family=family, model_type=model_type)
            for family, model_type in FAMILY_TYPES.items()
        ],
        ignore_index=True,
    )
    tables["latency_neural"] = pd.concat(
        [paired, paired.assign(device="cuda:0"), paired.assign(device="cuda:0", batch_size=128)],
        ignore_index=True,
    )
    tables["shared_latency_summary"]["pair_identity_json"] = json.dumps(
        PENN_IDENTITY, sort_keys=True
    )
    navigation = []
    navigation_methods = [(f, v) for f in ("tcn", "transformer") for v in ("default", "tuned")]
    navigation_methods += [(m, "classical") for m in ("zero", "train_mean", "ridge", "ridge_tuned")]
    for family, variant in navigation_methods:
        for index, metric in enumerate(tool.NAV_METRICS, 1):
            navigation.append(
                {
                    "task": "navigation",
                    "protocol_id": "navigation_rotation_0",
                    "split": "test",
                    "aggregation_level": "dataset",
                    "dataset_id": "smartpnt",
                    "low_sensor": "LOW",
                    "high_sensor": "HIGH",
                    "method": "model" if variant != "classical" else family,
                    "model_family": family,
                    "variant": variant,
                    "metric": metric,
                    "mean": float(index),
                    "sample_sd": 0.02 if variant != "classical" else np.nan,
                    "n": 3 if variant != "classical" else 1,
                    "unit": "m",
                }
            )
    tables["metrics"] = pd.concat([tables["metrics"], pd.DataFrame(navigation)], ignore_index=True)
    nav_latency = paired[paired.model_family.isin(["tcn", "transformer"])].assign(
        task="navigation", protocol_id="navigation_rotation_0"
    )
    tables["latency_neural"] = pd.concat([tables["latency_neural"], nav_latency], ignore_index=True)
    return tables


def selected_pages(tool, tables):
    pages = tool.plan_pages(tables, tool.default_config())
    chosen = []
    for prefix in ("shared_main_metrics_", "bank_supplement_metrics_"):
        chosen.extend(page["id"] for page in pages if page["id"].startswith(prefix))
    chosen.append(
        next(
            page["id"]
            for page in pages
            if page["kind"] == "metrics" and page["data"].task.eq("navigation").all()
        )
    )
    for device, batch in (("cpu", 1), ("cuda:0", 1), ("cuda:0", 128)):
        chosen.append(
            next(
                page["id"]
                for page in pages
                if page["kind"] == "latency"
                and not page["classical"]
                and page["data"].task.eq("paired").all()
                and page["data"].device.eq(device).all()
                and page["data"].batch_size.eq(batch).all()
            )
        )
    chosen.append(
        next(
            page["id"]
            for page in pages
            if page["kind"] == "latency"
            and not page["classical"]
            and page["data"].task.eq("navigation").all()
        )
    )
    for kind in ("shared_parameters", "shared_latency", "shared_workload"):
        chosen.append(next(page["id"] for page in pages if page["kind"] == kind))
    return chosen


def test_annotation_is_additive_and_keeps_values_ranks_variants_and_scopes():
    tool = module()
    data = pd.DataFrame(
        [
            {
                "task": "paired",
                "method": "model",
                "model_family": "gru",
                "model_type": "gru",
                "variant": "default",
                "information_scope": "protocol_unified_neural",
                "mean": 2.0,
                "rank": 7.5,
                "n": 3,
            },
            {
                "task": "paired",
                "method": "ridge_tuned",
                "model_family": "ridge_tuned",
                "model_type": None,
                "variant": "shared",
                "information_scope": "protocol_shared_classical",
                "mean": -0.2,
                "rank": 1.0,
                "n": 1,
            },
            {
                "task": "paired",
                "method": "model",
                "model_family": "plain_cnn",
                "model_type": "plain_cnn",
                "variant": "tuned",
                "information_scope": "protocol_unified_neural",
                "mean": 0.5,
                "rank": 2.0,
                "n": 3,
            },
        ]
    )
    result = tool.annotate_temporal_frame(data)
    pd.testing.assert_frame_equal(result[data.columns], data)
    assert result.temporal_label.tolist() == ["Causal", "Causal", "Window"]
    pd.testing.assert_frame_equal(tool.annotate_temporal_frame(result), result)
    assert result.temporal_note.str.contains(
        "not an end-to-end online guarantee", regex=False
    ).all()


def test_unknown_or_conflicting_model_identity_fails_and_high_is_reference():
    tool = module()
    with pytest.raises(ValueError, match="Unknown temporal task"):
        tool.annotate_temporal_frame(pd.DataFrame([{"task": "unknown", "model_family": "gru"}]))
    for family, model_type in (
        ("unknown", None),
        ("tcn_original", None),
        ("gru", "cnn_bilstm"),
        ("deepgyro_adapter", "deepgyro_adapter"),
    ):
        with pytest.raises(ValueError):
            tool.annotate_temporal_frame(
                pd.DataFrame(
                    [
                        {
                            "task": "paired",
                            "method": "model",
                            "model_family": family,
                            "model_type": model_type,
                        }
                    ]
                )
            )
    high = tool.annotate_temporal_frame(pd.DataFrame([{"task": "attitude", "method": "HIGH"}]))
    assert high.temporal_class.tolist() == ["measured_reference"]
    assert high.temporal_label.tolist() == ["Reference"]
    with pytest.raises(ValueError, match="variant"):
        tool.annotate_temporal_frame(pd.DataFrame([{"method": "gru/unknown"}]))
    display = pd.DataFrame([{"method": "deepgyro_adapter/tuned"}])
    result = tool.annotate_temporal_frame(
        display,
        protocol="penn_only",
        model_types={("penn_only", "deepgyro_adapter", "tuned"): "deepgyro_backbone"},
    )
    assert result.temporal_model_type.tolist() == ["deepgyro_backbone"]
    with pytest.raises(ValueError, match="model_type"):
        tool.annotate_temporal_frame(
            display,
            protocol="penn_only",
            model_types={("penn_only", "deepgyro_adapter", "tuned"): "gru"},
        )
    for row in (
        {
            "task": "paired",
            "protocol_id": "navigation_rotation_0",
            "model_family": "gru",
            "model_type": "gru",
        },
        {
            "task": "navigation",
            "protocol_id": "penn_only",
            "model_family": "gru",
            "model_type": "gru",
        },
        {"task": "navigation", "protocol_id": "navigation_rotation_0", "method": "HIGH"},
        {"task": "navigation", "method": "HIGH"},
    ):
        with pytest.raises(ValueError, match=r"disagree|paired reference"):
            tool.annotate_temporal_frame(pd.DataFrame([row]))


def test_mixed_groups_are_visible_and_original_global_ranks_remain_exact():
    tool = module()
    for width in (3.5, 7.2):
        figure, _ = tool._base(
            "Known-device bank (supplementary) · canonical Penn Tango Top IMU → ADIS16488 · "
            + "X" * 160,
            width=width,
            height=4.2,
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        box = figure._suptitle.get_window_extent(renderer)
        assert box.x0 >= 0 and box.x1 <= figure.bbox.width and box.y1 <= figure.bbox.height
        assert box.y0 > max(axis.get_window_extent(renderer).y1 for axis in figure.axes)
        tool.plt.close(figure)
        for xlabel in ("[m]", "Torch parameters", "P50 → P95 wall [ms]"):
            figure, axes = tool._base("Short navigation endpoint panel", width=width, height=3.5)
            axes[0, 0].plot([0, 1], [0, 1])
            axes[0, 0].set_xlabel(xlabel)
            figure.text(
                0.02,
                0.03,
                "Fixed representative; CPU/GPU and FP32/float64 scopes are separate.",
                fontsize=6.4,
            )
            figure.text(0.02, 0.075, tool.TEMPORAL_CONDITION, fontsize=6)
            figure.text(0.02, 0.112, "cpu; B=1 windows/call; W=80 samples.", fontsize=6.2)
            tool._reserve_footer(figure)
            figure.canvas.draw()
            footers, labels = tool._footer_axis_geometry(figure)
            assert not tool._footer_axis_overlaps(figure)
            assert min(box.y0 for _, box in labels) >= max(box.y1 for _, box in footers) + 7.5
            assert all(
                box.x0 >= 0
                and box.x1 <= figure.bbox.width
                and box.y0 >= 0
                and box.y1 <= figure.bbox.height
                for _, box in footers
            )
            tool.plt.close(figure)
    tables = fixture()
    original_ranks = tables["ranks"].copy(deep=True)
    pages = tool.plan_pages(tables, tool.default_config())
    ranks = next(page for page in pages if page["kind"] == "ranks")["data"]
    pd.testing.assert_frame_equal(ranks[original_ranks.columns], original_ranks)
    for page in pages:
        if page["kind"] in {"metrics", "deltas", "transfer"}:
            original = tables[
                "metrics_shared" if page["id"].startswith("shared_main_") else page["kind"]
            ]
            fields = [
                "protocol_id",
                "method",
                "model_family",
                "variant",
                "metric",
                "mean",
                "sample_sd",
                "n",
                "unit",
            ]
            fields = [field for field in fields if field in original and field in page["data"]]
            source = {tuple(row) for row in original[fields].fillna("<missing>").to_numpy()}
            assert {
                tuple(row) for row in page["data"][fields].fillna("<missing>").to_numpy()
            } <= source
    for identifier in selected_pages(tool, tables):
        page = next(page for page in pages if page["id"] == identifier)
        figure = tool.draw_page(page, tool.default_config())
        figure.canvas.draw()
        assert not tool._footer_axis_overlaps(figure)
        headers = {text.get_text() for axis in figure.axes for text in axis.texts}
        labels = set(page["data"].temporal_label)
        assert labels <= headers
        if page["kind"] == "metrics" and page["data"].task.eq("paired").all():
            assert {"Causal", "Window"} <= headers
            names = [text.get_text() for text in figure.axes[0].get_yticklabels()]
            if "TCN [D]" in names and "CNN [D]" in names:
                assert names.index("TCN [D]") < names.index("CNN [D]")
        if (
            page["data"].get("task", pd.Series(dtype=str)).eq("navigation").all()
            and "task" in page["data"]
        ):
            assert labels == {"Window endpoint"}
        tool.plt.close(figure)


def test_temporal_exports_bind_helper_source_columns_and_actual_group_headers(tmp_path):
    tool = module()
    tables = fixture()
    index = tool.build_synthetic(
        tables, tmp_path / "temporal", page_ids=selected_pages(tool, tables)
    )
    expected = tool.sha256_file(tool.TEMPORAL_SOURCE)
    assert index["temporal_support_sha256"] == expected
    snapshot = tmp_path / "temporal" / index["temporal_support_source"]["path"]
    assert tool.sha256_file(snapshot) == expected
    for entry in index["figures"]:
        assert entry["temporal_support_sha256"] == expected
        source = pd.read_csv(tmp_path / "temporal" / entry["source_data"][0]["path"])
        assert set(tool.TEMPORAL_FIELDS) <= set(source)
        qa = json.loads((tmp_path / "temporal" / entry["qa"]["path"]).read_text())
        assert qa["temporal_support_sha256"] == expected
        assert qa["temporal_grouping"]["required"]
        assert set(source.temporal_label) <= set(qa["temporal_grouping"]["visible_headers"])
        assert set(entry["files"]) == {"svg", "pdf", "png", "tiff"}
