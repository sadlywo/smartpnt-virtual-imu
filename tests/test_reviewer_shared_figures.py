"""Synthetic-only shared figure contracts, exact support joins and exports."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from test_reviewer_figures import extended_fixture, module

PENN_IDENTITY = {
    "dataset_id": "penncosyvio",
    "low_sensor": "Tango Top IMU",
    "high_sensor": "ADIS16488",
}
AL_IDENTITIES = [
    {"dataset_id": "alimu", "low_sensor": sensor, "high_sensor": "VN100"}
    for sensor in ("BMX160", "MPU9250", "LSM9DS1")
]


def fixture():
    tool = module()
    import reviewer_shared_figure_data as shared

    tables = extended_fixture()
    classical = tables["metrics"].query("variant == 'classical'").copy()
    raw = classical.rename(
        columns={
            "protocol_id": "protocol",
            "method": "canonical_method",
            "aggregation_level": "metric_scope",
            "mean": "value",
            "sample_sd": "sd",
        }
    )
    raw["information_scope"], raw["variant"] = shared.SCOPE, "shared"
    raw["value"] *= 1.2  # Deliberately worse than the old bank; never select it away.
    tables["metrics_shared"] = shared.main_metrics(tables["metrics"], raw)
    downstream = tables["downstream"]
    raw_down = downstream[downstream.method.isin(tool.CLASSICAL)].copy()
    raw_down["canonical_method"] = raw_down.method
    raw_down["information_scope"], raw_down["variant"] = shared.SCOPE, "shared"
    raw_down["n"], raw_down["sd"], raw_down["status"] = 1, np.nan, "supported"
    for stem in ("endpoint_rmse_deg", "position_endpoint_rmse_m"):
        raw_down[stem] = raw_down[stem + "_mean"] * 1.2
    tables["downstream_shared"] = shared.main_downstream(downstream, raw_down)
    unit = copy.deepcopy(tables["temporal"][0])
    for column, value in (
        ("dataset_id", "penncosyvio"),
        ("low_sensor", "LOW"),
        ("high_sensor", "HIGH"),
    ):
        unit["frame"][column] = value
    saved = unit["frame"][[*shared.PHYSICAL_KEYS, *tool.LOW, *tool.HIGH]].copy()
    for axis in tool.AXES:
        for method in tool.CLASSICAL:
            saved[method + "_" + axis] = unit["frame"][axis + "_low"] + (
                0 if method == "identity" else 0.03
            )
    source = {
        "path": "synthetic/aux3/signal_evaluation/penn_only/test_predictions.parquet",
        "sha256": "b" * 64,
    }
    tables["temporal_shared"] = [shared.join_shared_temporal(unit, saved, source)]
    state, totals, latency, aliases = [], [], [], []
    for method in tool.CLASSICAL:
        aliases.append(
            {
                "protocol": "penn_only",
                "information_scope": shared.SCOPE,
                "canonical_method": method,
                "variant": "shared",
                "alias_status": "common_identity" if method == "identity" else "not_alias",
            }
        )
        for scope, variant, count in ((shared.SCOPE, "shared", 42), (shared.BANK, "bank", 84)):
            count = 0 if method == "identity" else count
            common = {
                "protocol": "penn_only",
                "information_scope": scope,
                "canonical_method": method,
                "variant": variant,
            }
            state.append(
                {
                    **common,
                    "coefficient_count": count,
                    "coefficient_bytes": count * 8,
                    "serialized_state_bytes": 1100 + count,
                    "normalizer_count": 0,
                    "normalizer_bytes": 0,
                    "pair_identity_json": "{}",
                }
            )
            totals.append(
                {
                    **common,
                    "bank_coefficient_count": count,
                    "bank_coefficient_bytes": count * 8,
                    "all_artifact_serialized_bytes": 1100 + count,
                    "numeric_unique_coefficient_count": count,
                    "numeric_unique_coefficient_bytes": count * 8,
                }
            )
            latency.append(
                {
                    **common,
                    "pair_identity_json": "{}",
                    "batch_size_per_pair": 1,
                    "window_size": 80,
                    "pair_count": 1,
                    "windows_per_call": 1,
                    "p50_ms": 0.11,
                    "p95_ms": 0.23,
                    "loop_throughput_windows_per_second": 7000,
                    "scope": "float64_algorithm_forward",
                    "dtype": "float64",
                }
            )
    tables.update(
        shared_state_inventory=pd.DataFrame(state),
        shared_bank_parameter_totals=pd.DataFrame(totals),
        shared_latency_summary=pd.DataFrame(latency),
        shared_workload_latency=pd.DataFrame(latency).assign(pair_count=4, windows_per_call=4),
        shared_alias_audit=pd.DataFrame(aliases),
        _analysis_profile=shared.PROFILE,
        _shared_controls_index_sha256="c" * 64,
        _shared_coverage={**shared.COUNTS, "neural_split_support_references": 720},
    )
    tables["_sources"] = {
        name: [{"path": "synthetic/aux3/" + name + ".csv", "sha256": "d" * 64}]
        for name in (
            "metrics_shared",
            "downstream_shared",
            "shared_state_inventory",
            "shared_bank_parameter_totals",
            "shared_latency_summary",
            "shared_latency_raw",
            "shared_workload_latency",
            "shared_alias_audit",
        )
    }
    return tables, unit, saved


def realistic_cost_fixture(protocol, identities):
    tables, _, _ = fixture()
    original = tables["shared_latency_summary"]
    tables["shared_latency_summary"] = pd.concat(
        [
            original.assign(
                protocol=protocol,
                pair_identity_json=json.dumps(identity, sort_keys=True),
                batch_size_per_pair=128,
                window_size=320,
                windows_per_call=128,
            )
            for identity in identities
        ],
        ignore_index=True,
    )
    return tables


@pytest.mark.parametrize(
    "protocol,identities",
    [
        ("penn_only", [PENN_IDENTITY]),
        ("alimu_only", [AL_IDENTITIES[-1]]),
        ("pooled", [*AL_IDENTITIES, PENN_IDENTITY]),
    ],
)
def test_canonical_pair_titles_fit_without_changing_physical_groups(protocol, identities):
    tool = module()
    tables = realistic_cost_fixture(protocol, identities)
    pages = [
        page
        for page in tool.plan_pages(tables, tool.default_config())
        if page["kind"] == "shared_latency"
    ]
    assert len(pages) == len(identities)
    expected = {json.dumps(identity, sort_keys=True) for identity in identities}
    assert {
        page["coverage"]["source_group_key"]["pair_identity_json"] for page in pages
    } == expected
    for page in pages:
        assert "pair_identity_json" in page["data"]
        assert page["data"].pair_identity_json.nunique() == 1
        assert page["data"].p50_ms.eq(0.11).all()
        assert "dataset_id" not in page["title"] and "→" in page["title"]
        figure = tool.draw_page(page, tool.default_config())
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        title_box = figure._suptitle.get_window_extent(renderer)
        assert title_box.x0 >= 0 and title_box.x1 <= figure.bbox.width
        assert title_box.y0 > max(axis.get_window_extent(renderer).y1 for axis in figure.axes)
        tool.plt.close(figure)


def test_canonical_penn_title_exports_all_four_formats_with_raw_identity(tmp_path):
    tool = module()
    tables = realistic_cost_fixture("penn_only", [PENN_IDENTITY])
    page = next(
        page
        for page in tool.plan_pages(tables, tool.default_config())
        if page["kind"] == "shared_latency"
    )
    index = tool.build_synthetic(tables, tmp_path / "canonical_penn", page_ids=[page["id"]])
    entry = index["figures"][0]
    assert set(entry["files"]) == {"svg", "pdf", "png", "tiff"}
    source = pd.read_csv(tmp_path / "canonical_penn" / entry["source_data"][0]["path"])
    assert set(source.pair_identity_json) == {json.dumps(PENN_IDENTITY, sort_keys=True)}
    assert "Tango Top IMU → ADIS16488" in entry["title"]
    assert entry["qa"]["status"] == "passed"


def test_shared_and_bank_scopes_never_average_or_become_replicates():
    tool = module()
    import reviewer_shared_figure_data as shared

    tables, _, _ = fixture()
    pages = tool.plan_pages(tables, tool.default_config())
    main = [page for page in pages if page["id"].startswith("shared_main_")]
    assert main and all(page["coverage"]["information_scope"] == shared.SCOPE for page in main)
    accuracy = next(page for page in main if page["kind"] == "metrics")
    data = accuracy["data"]
    assert not data.information_scope.eq(shared.BANK).any()
    assert data.loc[data.variant.eq("shared"), "n"].eq(1).all()
    assert data.loc[data.variant.eq("shared"), "sample_sd"].isna().all()
    assert len(data[data.method.eq("identity")]) == 6
    assert not data.duplicated(["model_family", "variant", "metric"]).any()
    bank = next(page for page in pages if page["id"].startswith("bank_supplement_metrics"))
    assert "Known-device bank" in bank["title"]
    assert tool.label(data[data.method.eq("affine")].iloc[0]) != tool.label(
        bank["data"].query("method == 'affine'").iloc[0]
    )
    downstream = next(page for page in main if page["kind"] == "downstream" and page["gains"])
    assert downstream["data"].gain_percent.min() < 0
    gain_figure = tool.draw_page(downstream, tool.default_config())
    assert all("[%]" in axis.get_title() for axis in gain_figure.axes[:2])
    tool.plt.close(gain_figure)
    assert all(page.get("figure_contract") for page in pages)
    assert sum(page["kind"] == "training" for page in pages) == 2
    integrals = next(page for page in main if page["kind"] == "segment_integrals")["data"]
    identity = integrals[integrals.method.eq("identity")]
    assert len(identity) == 4 and not identity.duplicated(["sequence_id", "sensor"]).any()
    zero_reference = copy.deepcopy(tables)
    zero_reference["downstream_shared"].loc[
        zero_reference["downstream_shared"].method.eq("identity"),
        ["endpoint_rmse_deg_mean", "position_endpoint_rmse_m_mean"],
    ] = 0
    zero_pages = tool.plan_pages(zero_reference, tool.default_config())
    undefined = next(
        page
        for page in zero_pages
        if page["id"].startswith("shared_main_") and page["kind"] == "downstream" and page["gains"]
    )["data"]
    assert undefined.gain_percent.isna().all()
    assert set(undefined.gain_status) == {"unsupported_near_zero_identity_rmse"}


def test_mixed_scopes_wrong_n_missing_units_and_duplicate_identity_reject():
    tool = module()
    import reviewer_shared_figure_data as shared

    tables, _, _ = fixture()
    raw = (
        tables["metrics_shared"]
        .query("variant == 'shared'")
        .rename(
            columns={
                "protocol_id": "protocol",
                "method": "canonical_method",
                "aggregation_level": "metric_scope",
                "mean": "value",
                "sample_sd": "sd",
            }
        )
    )
    for column, value in (("information_scope", shared.BANK), ("n", 3), ("sd", 0.0)):
        bad = raw.copy()
        bad.loc[bad.index[0], column] = value
        with pytest.raises(ValueError):
            shared.main_metrics(tables["metrics"], bad)
    with pytest.raises(ValueError, match="repeat"):
        shared.main_metrics(tables["metrics"], pd.concat([raw, raw.iloc[:1]]))
    with pytest.raises(ValueError, match="support"):
        shared.main_metrics(tables["metrics"], raw.assign(low_sensor="WRONG"))
    tables["_shared_coverage"]["neural_split_support_references"] = 60
    with pytest.raises(ValueError, match="coverage"):
        tool.plan_pages(tables, tool.default_config())


def test_saved_temporal_exact_keys_values_and_identity_once():
    import reviewer_shared_figure_data as shared

    _, unit, saved = fixture()
    joined = shared.join_shared_temporal(unit, saved.iloc[::-1], {})
    assert all(meta["method"] != "identity" for meta in joined["methods"].values())
    assert len(joined["methods"]) == 9  # Eight shared learned methods and one synthetic neural.
    for column in ("gx_high", "gx_low", "timestamp_s", "identity_gx"):
        bad = saved.copy()
        bad.loc[0, column] += 1
        with pytest.raises(ValueError):
            shared.join_shared_temporal(unit, bad, {})
    with pytest.raises(ValueError, match="duplicated"):
        shared.join_shared_temporal(unit, pd.concat([saved, saved.iloc[:1]]), {})


def writer_shaped_temporal_fixture():
    """Small synthetic values with the authenticated saved writer's column names."""
    import reviewer_shared_figure_data as shared

    frame = pd.DataFrame(
        {
            "dataset_id": ["penncosyvio"] * 4,
            "low_sensor": ["Tango Top IMU"] * 4,
            "high_sensor": ["ADIS16488"] * 4,
            "sequence_id": ["bs_seg0", "bs_seg0", "bs_seg1", "bs_seg1"],
            "timestamp_s": [0.0, 0.025, 0.0, 0.025],
            "source_row": [10, 11, 20, 21],
            "source_timestamp_s": [1.0, 1.025, 2.0, 2.025],
            "recording_id": ["bs"] * 4,
            "segment_id": ["bs_seg0", "bs_seg0", "bs_seg1", "bs_seg1"],
            "pair_id": ["Tango_Top_IMU_to_ADIS16488"] * 4,
        },
        index=[8, 3, 12, 5],
    )
    for index, axis in enumerate(shared.AXES):
        frame[axis + "_low"] = np.arange(4, dtype=float) + (index + 1) / 8
        frame[axis + "_high"] = frame[axis + "_low"] + 0.5
        frame[axis + "_neural_default"] = frame[axis + "_low"] + 0.75
        frame[axis + "_neural_tuned"] = frame[axis + "_low"] + 0.875
    unit = {
        "protocol_id": "pooled",
        "identity": "Penn_bs",
        "frame": frame,
        "methods": {
            "neural_default": {
                "method": "gru/default",
                "family": "gru",
                "variant": "default",
                "seed": 20261010,
            },
            "neural_tuned": {
                "method": "gru/tuned",
                "family": "gru",
                "variant": "tuned",
                "seed": 20261010,
            },
            "classic_bias_mean": {"method": "bias_mean", "variant": "fixed", "seed": None},
        },
        "source_tables": [
            {"path": "synthetic/original/evaluated_signals.parquet", "sha256": "a" * 64}
        ],
    }
    saved = frame.drop(
        columns=[
            axis + "_" + prefix
            for axis in shared.AXES
            for prefix in ("neural_default", "neural_tuned")
        ]
    ).copy()
    for index, method in enumerate(shared.METHODS):
        for axis in shared.AXES:
            saved[method + "_" + axis] = saved[axis + "_low"] + index / 16
    return unit, saved


def test_saved_temporal_writer_schema_superset_preserves_order_neural_and_sources():
    import reviewer_shared_figure_data as shared

    unit, saved = writer_shaped_temporal_fixture()
    before = copy.deepcopy(unit)
    saved_before = saved.copy(deep=True)
    extra = saved.iloc[:1].copy()
    extra["dataset_id"], extra["low_sensor"], extra["high_sensor"] = "alimu", "BMX160", "VN100"
    source = {"path": "synthetic/shared/pooled/test_predictions.parquet", "sha256": "b" * 64}
    joined = shared.join_shared_temporal(unit, pd.concat([extra, saved.iloc[::-1]]), source)
    assert set(shared.AXES).isdisjoint(saved)
    pd.testing.assert_frame_equal(joined["frame"][before["frame"].columns], before["frame"])
    pd.testing.assert_frame_equal(unit["frame"], before["frame"])
    pd.testing.assert_frame_equal(saved, saved_before)
    assert unit["methods"] == before["methods"]
    assert joined["source_tables"] == [*before["source_tables"], source]
    assert unit["source_tables"] == before["source_tables"]
    assert joined["information_scope"] == shared.SCOPE
    assert len(joined["methods"]) == 10  # Two original neural + eight shared learned.
    for prefix in ("neural_default", "neural_tuned"):
        assert joined["methods"][prefix] == before["methods"][prefix]
    for method in shared.METHODS:
        if method == "identity":
            assert "shared_identity" not in joined["methods"]
            continue
        prefix = "shared_" + method
        assert joined["methods"][prefix] == {
            "method": method + " [shared]",
            "family": method,
            "variant": "shared",
            "seed": None,
            "information_scope": shared.SCOPE,
        }
        for axis in shared.AXES:
            np.testing.assert_array_equal(
                joined["frame"][axis + "_" + prefix], saved[method + "_" + axis]
            )


@pytest.mark.parametrize(
    "column", ["dataset_id", "low_sensor", "high_sensor", "sequence_id", "timestamp_s"]
)
@pytest.mark.parametrize("side", ["original", "shared"])
def test_saved_temporal_missing_physical_key_rejects(column, side):
    import reviewer_shared_figure_data as shared

    unit, saved = writer_shaped_temporal_fixture()
    if side == "original":
        unit["frame"] = unit["frame"].drop(columns=column)
    else:
        saved = saved.drop(columns=column)
    with pytest.raises(ValueError, match="physical keys"):
        shared.join_shared_temporal(unit, saved, {})


@pytest.mark.parametrize(
    "column", ["dataset_id", "low_sensor", "high_sensor", "sequence_id", "timestamp_s"]
)
def test_saved_temporal_changed_physical_key_rejects(column):
    import reviewer_shared_figure_data as shared

    unit, saved = writer_shaped_temporal_fixture()
    saved = saved.copy()
    row = saved.index[0]
    saved.loc[row, column] = (
        np.nextafter(saved.loc[row, column], np.inf) if column == "timestamp_s" else "WRONG"
    )
    with pytest.raises(ValueError, match="support differs"):
        shared.join_shared_temporal(unit, saved, {})


@pytest.mark.parametrize("axis", ["gx", "gy", "gz", "ax", "ay", "az"])
@pytest.mark.parametrize("reference", ["low", "high"])
def test_saved_temporal_one_ulp_physical_value_rejects(axis, reference):
    import reviewer_shared_figure_data as shared

    unit, saved = writer_shaped_temporal_fixture()
    column, row = axis + "_" + reference, saved.index[0]
    saved.loc[row, column] = np.nextafter(saved.loc[row, column], np.inf)
    with pytest.raises(ValueError, match="physical values/metadata differ"):
        shared.join_shared_temporal(unit, saved, {})


@pytest.mark.parametrize("axis", ["gx", "gy", "gz", "ax", "ay", "az"])
def test_saved_temporal_missing_high_rejects_even_with_bare_axes(axis):
    import reviewer_shared_figure_data as shared

    unit, saved = writer_shaped_temporal_fixture()
    for name in shared.AXES:
        saved[name] = saved[name + "_high"]
    saved = saved.drop(columns=axis + "_high")
    with pytest.raises(ValueError, match="physical keys"):
        shared.join_shared_temporal(unit, saved, {})


@pytest.mark.parametrize("side", ["original", "shared"])
def test_saved_temporal_duplicate_or_absent_support_rejects(side):
    import reviewer_shared_figure_data as shared

    unit, saved = writer_shaped_temporal_fixture()
    if side == "original":
        unit["frame"] = pd.concat([unit["frame"], unit["frame"].iloc[:1]])
    else:
        saved = pd.concat([saved, saved.iloc[:1]])
    with pytest.raises(ValueError, match="duplicated"):
        shared.join_shared_temporal(unit, saved, {})
    unit, saved = writer_shaped_temporal_fixture()
    with pytest.raises(ValueError, match="support differs"):
        shared.join_shared_temporal(unit, saved.iloc[1:], {})


@pytest.mark.parametrize(
    "column", ["source_row", "source_timestamp_s", "recording_id", "segment_id", "pair_id"]
)
def test_saved_temporal_common_audit_metadata_rejects(column):
    import reviewer_shared_figure_data as shared

    unit, saved = writer_shaped_temporal_fixture()
    row = saved.index[0]
    saved.loc[row, column] = (
        saved.loc[row, column] + 1 if column.startswith("source_") else "WRONG"
    )
    with pytest.raises(ValueError, match="physical values/metadata differ"):
        shared.join_shared_temporal(unit, saved, {})


@pytest.mark.parametrize("axis", ["gx", "gy", "gz", "ax", "ay", "az"])
def test_saved_temporal_identity_must_equal_low_exactly(axis):
    import reviewer_shared_figure_data as shared

    unit, saved = writer_shaped_temporal_fixture()
    column, row = "identity_" + axis, saved.index[0]
    saved.loc[row, column] = np.nextafter(saved.loc[row, column], np.inf)
    with pytest.raises(ValueError, match="identity is not the exact"):
        shared.join_shared_temporal(unit, saved, {})


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_saved_temporal_nonfinite_prediction_rejects(value):
    import reviewer_shared_figure_data as shared

    unit, saved = writer_shaped_temporal_fixture()
    saved.loc[saved.index[0], "bias_mean_gx"] = value
    with pytest.raises(ValueError, match="nonfinite"):
        shared.join_shared_temporal(unit, saved, {})


def test_source_csv_units_keep_axis_gyro_accel_counts_and_relative_values_distinct():
    import reviewer_shared_figure_data as shared

    assert shared.metric_unit("axis_rmse_gx") == "rad/s"
    assert shared.metric_unit("axis_mean_error_gy") == "rad/s"
    assert shared.metric_unit("axis_mae_ax") == "m/s^2"
    assert shared.metric_unit("samples") == "samples"
    assert shared.metric_unit("gyro_vector_rmse_radps_relative_identity") == "dimensionless"
    with pytest.raises(ValueError, match="unit"):
        shared.metric_unit("unknown_metric")


def test_shared_profile_requires_both_receipts_and_aux3_roots():
    import reviewer_shared_figure_data as shared

    class Package:
        def __init__(self):
            self.manifest = {"analysis_profile": shared.PROFILE}
            self.roots = {"aux3", "aux3_project"}
            self.inner = shared.PROFILE

        def read_json(self, _):
            return {"analysis_profile": self.inner}

    package = Package()
    assert shared.analysis_profile(package) == shared.PROFILE
    package.roots = {"aux3"}
    with pytest.raises(ValueError, match="roots"):
        shared.analysis_profile(package)
    package.roots = {"aux3", "aux3_project"}
    for inner in (None, "unknown"):
        package.inner = inner
        with pytest.raises(ValueError, match="profile"):
            shared.analysis_profile(package)


def test_public_shared_loader_uses_full_semantic_gate_and_source_hashes(tmp_path):
    tool = module()
    import reviewer_shared_figure_data as shared
    from test_reviewer_shared_report import shared_fixture

    package_path, _ = shared_fixture(tmp_path)
    package = tool.load_package(package_path, allow_synthetic=True)
    roots = {"aux3": package.root_path("aux3")}
    state = package.read_json(package.root_path("core_pipeline") / "pipeline.json")
    finals = [job for job in state["jobs"].values() if job["phase"].startswith("final_")]
    core_sha = package.verify_file(package.root_path("core_pipeline") / "pipeline.json")
    reader = tool.SourceReader(package)
    result = shared.load_shared(reader, roots, finals, core_sha)
    assert len(result["frames"]["signal_registry"]) == 108
    assert len(result["frames"]["neural_support_registry"]) == 720
    assert reader.sources
    target = roots["aux3"] / result["descriptors"]["metrics_long"]["path"]
    target.write_bytes(target.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        shared.load_shared(reader, roots, finals, core_sha)


@pytest.mark.parametrize("mutation", ["missing_index", "index_profile", "index_coverage"])
def test_missing_or_signed_wrong_shared_index_rejects(tmp_path, mutation):
    tool = module()
    import reviewer_shared_figure_data as shared
    from test_reviewer_shared_report import shared_fixture

    package_path, _ = shared_fixture(tmp_path)
    manifest_path = package_path / "metadata/package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index_path = package_path / manifest["roots"]["aux3"] / "shared_controls_artifacts.json"
    if mutation == "missing_index":
        index_path.unlink()
    else:
        value = json.loads(index_path.read_text(encoding="utf-8"))
        if mutation == "index_profile":
            value["analysis_profile"] = "wrong_profile"
        else:
            value["counts"]["shared_logical_controls"] = 53
        index_path.write_text(json.dumps(value), encoding="utf-8")
        member = index_path.relative_to(package_path).as_posix()
        manifest["files"][member] = tool.sha256_file(index_path)
        manifest["file_metadata"][member]["size_bytes"] = index_path.stat().st_size
        audit_path = package_path / "metadata/completion_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["shared_controls_index"]["sha256"] = tool.sha256_file(index_path)
        audit["auxiliary_checks"]["aux3"] = value
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        manifest["files"]["metadata/completion_audit.json"] = tool.sha256_file(audit_path)
        manifest["file_metadata"]["metadata/completion_audit.json"]["size_bytes"] = (
            audit_path.stat().st_size
        )
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    package = tool.load_package(package_path, allow_synthetic=True, verify=False)
    state = package.read_json(package.root_path("core_pipeline") / "pipeline.json")
    finals = [job for job in state["jobs"].values() if job["phase"].startswith("final_")]
    with pytest.raises(ValueError, match=r"missing|profile/counts"):
        shared.load_shared(
            tool.SourceReader(package),
            {"aux3": package.root_path("aux3")},
            finals,
            package.verify_file(package.root_path("core_pipeline") / "pipeline.json"),
        )


def test_unexcited_coherence_is_explicitly_unsupported_without_fabrication():
    tool = module()
    tables, _, _ = fixture()
    unit = copy.deepcopy(tables["temporal_shared"][0])
    unit["frame"]["gx_low"] = 0
    unit["frame"]["gx_high"] = 0
    pages = tool._temporal_pages(unit, tool.default_config())
    spectral = next(page for page in pages if page["kind"] == "spectral")
    axis = spectral["data"].query("axis == 'gx'")
    assert axis.low_coherence.isna().all()
    assert axis.prediction_coherence.isna().all()
    assert set(axis.low_coherence_status) == {"unsupported_unexcited_axis"}
    figure = tool.draw_page(spectral, tool.default_config())
    figure.canvas.draw()
    tool.plt.close(figure)


def test_new_shared_draw_branches_export_four_formats_and_bind_qa(tmp_path):
    tool = module()
    tables, _, _ = fixture()
    pages = tool.plan_pages(tables, tool.default_config())
    kinds = [
        "metrics",
        "downstream",
        "signals",
        "spectral",
        "segment_integrals",
        "shared_parameters",
        "shared_latency",
        "shared_workload",
        "shared_alias",
    ]
    chosen = [
        next(
            page["id"]
            for page in pages
            if page["kind"] == kind
            and page["coverage"].get("information_scope") == "protocol_shared_classical"
        )
        for kind in kinds
    ]
    # Exercise both per-state and deployment totals, plus the bank-state layout.
    chosen += [
        page["id"]
        for page in pages
        if page["kind"] == "shared_parameters" and page["id"] not in chosen
    ]
    shared_arx = next(
        page
        for page in pages
        if page["kind"] == "signals"
        and page["id"].startswith("shared_main_")
        and page["data"].method.eq("arx_tuned [shared]").any()
    )
    chosen.append(shared_arx["id"])
    index = tool.build_synthetic(tables, tmp_path / "figures", page_ids=chosen)
    assert index["analysis_profile"] == "reviewer_shared_classical_v1"
    assert index["shared_controls_index_sha256"] == "c" * 64
    assert index["shared_coverage"]["neural_split_support_references"] == 720
    assert index["qa_summary"]["visual_review_status"] == "pending"
    for entry in index["figures"]:
        assert set(entry["files"]) == {"svg", "pdf", "png", "tiff"}
        qa = json.loads((tmp_path / "figures" / entry["qa"]["path"]).read_text())
        assert qa["analysis_profile"] == index["analysis_profile"]
        assert qa["shared_controls_index_sha256"] == "c" * 64
        assert qa["pdf_actual_render_checked"] and qa["svg_editable_text"]
    changed = copy.deepcopy(tables)
    changed["_shared_controls_index_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="contract"):
        tool.build_synthetic(changed, tmp_path / "figures", page_ids=chosen, resume=True)
