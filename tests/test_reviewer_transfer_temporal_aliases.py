"""Producer-shaped transfer display aliases; no rendering or scientific execution."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from test_reviewer_figures import module

TRANSFER_ALIASES = {
    "source_bias_mean_fixed": "bias_mean",
    "source_affine_fixed": "affine",
    "source_ridge_affine_fixed": "ridge_affine",
    "source_learned_fir_causal_fixed": "learned_fir_causal",
    "source_arx_linear_fixed": "arx_linear",
    "source_ridge_tuned": "ridge_tuned",
    "source_fir_tuned": "fir_tuned",
    "source_arx_tuned": "arx_tuned",
}
FACTORY_TYPES = {
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
    "tartanimu_window_adapter": "tartanimu_window",
}
DIRECTIONS = {"alimu_to_penn": "alimu_only", "penn_to_alimu": "penn_only"}


@pytest.fixture(scope="module")
def figure_tool():
    return module()


def transfer_row(method, direction="alimu_to_penn"):
    return {
        "task": "paired",
        "protocol_id": direction,
        "split": "test",
        "dataset_id": "penncosyvio",
        "low_sensor": "Tango Top IMU",
        "high_sensor": "ADIS16488",
        "method": method,
        "model_family": method,
        "variant": "classical",
        "information_scope": "source_domain_shared",
        "metric": "gyro_vector_rmse_radps",
        "mean": 0.125,
        "sample_sd": np.nan,
        "reference_mean": np.nan,
        "rank": 7.5,
        "n": 1,
        "unit": "rad/s",
    }


@pytest.mark.parametrize("alias,canonical", TRANSFER_ALIASES.items())
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_exact_transfer_alias_is_classification_only(figure_tool, alias, canonical, direction):
    from reviewer_temporal_support import classify_temporal_support

    original = pd.DataFrame([transfer_row(alias, direction)], index=[19])
    before = original.copy(deep=True)
    result = figure_tool.annotate_temporal_frame(original)
    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_frame_equal(result[before.columns], before)
    expected = classify_temporal_support("paired", canonical)
    assert {key: result.iloc[0][key] for key in figure_tool.TEMPORAL_FIELDS} == expected
    assert result.iloc[0].temporal_class == "sample_causal_eval"
    assert result.iloc[0].temporal_model_type is None
    pd.testing.assert_frame_equal(figure_tool.annotate_temporal_frame(result), result)


def transfer_records(figure_tool):
    """Synthetic scalar metrics with all actual producer names and roster dimensions."""
    records = []
    for direction in DIRECTIONS:
        targets = (
            [("penncosyvio", "Tango Top IMU", "ADIS16488")]
            if direction == "alimu_to_penn"
            else [("alimu", sensor, "VN100") for sensor in ("BMX160", "MPU9250", "LSM9DS1")]
        )
        for family, factory in FACTORY_TYPES.items():
            if family == "tartanimu_window_adapter":
                continue
            for variant in ("default", "tuned"):
                for seed in (20261010, 20261011, 20261012):
                    pairs, references = [], []
                    for dataset, low, high in targets:
                        identity = {"dataset_id": dataset, "low_sensor": low, "high_sensor": high}
                        measurements = {
                            method: {
                                metric: (index + 1) / 8
                                for metric in figure_tool.PAIRED_METRICS
                            }
                            for index, method in enumerate(["identity", *TRANSFER_ALIASES])
                        }
                        measurements["model"] = {
                            metric: 1 + (seed - 20261010) / 10
                            for metric in figure_tool.PAIRED_METRICS
                        }
                        pairs.append({**identity, "metrics": measurements})
                        references.append(
                            {
                                **identity,
                                "metrics": {
                                    "model": {metric: 0.5 for metric in figure_tool.PAIRED_METRICS}
                                },
                            }
                        )
                    records.append(
                        {
                            "direction": direction,
                            "model_family": family,
                            "source_model_config": {"type": factory},
                            "variant": variant,
                            "seed": seed,
                            "target_fingerprint": direction,
                            "classical_state_sha256": direction,
                            "metrics": {"pairs": pairs},
                            "target_domain_reference": {"test": {"pairs": references}},
                        }
                    )
    return records


def test_transfer_table_all_actual_names_and_factory_lookup_preserve_rows(figure_tool):
    records = transfer_records(figure_tool)
    before_records = copy.deepcopy(records)
    assert len(records) == 120
    assert sum(len(record["metrics"]["pairs"]) for record in records) == 240
    frame = figure_tool._transfer_table(records)
    frame.loc[frame.method.ne("model"), "information_scope"] = "source_domain_shared"
    before = frame.copy(deep=True)
    model_types = {
        (protocol, family, variant): factory
        for protocol in DIRECTIONS.values()
        for family, factory in FACTORY_TYPES.items()
        if family != "tartanimu_window_adapter"
        for variant in ("default", "tuned")
    }
    result = figure_tool.annotate_temporal_frame(frame, model_types=model_types)
    pd.testing.assert_frame_equal(result[before.columns], before)
    pd.testing.assert_frame_equal(frame, before)
    assert records == before_records
    assert set(result.method) == {"model", "identity", *TRANSFER_ALIASES}
    classical = result[result.method.ne("model")]
    assert classical.n.eq(1).all() and classical.sample_sd.isna().all()
    assert classical.temporal_class.eq("sample_causal_eval").all()
    assert classical.information_scope.eq("source_domain_shared").all()
    neural = result[result.method.eq("model")]
    assert neural.n.eq(3).all() and neural.sample_sd.notna().all()
    assert neural.apply(
        lambda row: row.temporal_model_type == FACTORY_TYPES[row.model_family], axis=1
    ).all()
    for family, rows in neural.groupby("model_family"):
        expected = (
            "sample_causal_eval"
            if family in {"tcn", "gru", "deepgyro_adapter"}
            else "window_lookahead_eval"
        )
        assert rows.temporal_class.eq(expected).all()


@pytest.mark.parametrize(
    "overrides",
    [
        {"task": "navigation", "protocol_id": "navigation_rotation_0"},
        {"protocol_id": "penn_only"},
        {"protocol_id": "pooled"},
        {"protocol_id": "unknown_transfer"},
        {"variant": "default"},
        {"variant": "tuned"},
        {"variant": "shared"},
        {"variant": "fixed"},
        {"variant": None},
    ],
)
def test_transfer_alias_rejects_wrong_task_protocol_or_phase(figure_tool, overrides):
    # ridge_tuned is a valid canonical navigation method: scope must reject this alias first.
    row = {**transfer_row("source_ridge_tuned"), **overrides}
    with pytest.raises(ValueError):
        figure_tool.annotate_temporal_frame(pd.DataFrame([row]))


@pytest.mark.parametrize("alias", TRANSFER_ALIASES)
def test_transfer_classical_alias_rejects_neural_factory_type(figure_tool, alias):
    row = {**transfer_row(alias), "model_type": "gru"}
    with pytest.raises(ValueError):
        figure_tool.annotate_temporal_frame(pd.DataFrame([row]))


@pytest.mark.parametrize(
    "method",
    [
        "source_identity_fixed",
        "source_bias_mean",
        "source_affine",
        "source_ridge_affine_tuned",
        "source_learned_fir_causal_tuned",
        "source_gru_fixed",
        "source_unknown_fixed",
        "source_source_bias_mean_fixed",
        "source_bias_mean_fixed_extra",
    ],
)
def test_unregistered_transfer_like_names_are_not_generically_stripped(figure_tool, method):
    with pytest.raises(ValueError):
        figure_tool.annotate_temporal_frame(pd.DataFrame([transfer_row(method)]))


@pytest.mark.parametrize(
    "method,factory",
    [
        *FACTORY_TYPES.items(),
        *((name, None) for name in ("zero", "train_mean", "ridge", "ridge_tuned")),
    ],
)
def test_all_navigation_factories_and_baselines_remain_window_endpoint(
    figure_tool, method, factory
):
    row = {
        "task": "navigation",
        "protocol_id": "navigation_rotation_0",
        "method": "model" if factory else method,
        "model_family": method,
        "model_type": factory,
        "variant": "default" if factory else "classical",
        "mean": 0.5,
        "n": 3 if factory else 1,
    }
    frame = pd.DataFrame([row])
    result = figure_tool.annotate_temporal_frame(frame)
    pd.testing.assert_frame_equal(result[frame.columns], frame)
    assert result.iloc[0].temporal_class == "window_endpoint"
    assert result.iloc[0].temporal_model_type == factory


@pytest.mark.parametrize("alias", TRANSFER_ALIASES)
def test_core_classifier_still_rejects_display_source_aliases(alias):
    from reviewer_temporal_support import classify_temporal_support

    with pytest.raises(ValueError):
        classify_temporal_support("paired", alias)


def test_report_projection_route_does_not_gain_figure_only_aliases():
    from build_reviewer_report import temporal_row

    with pytest.raises(ValueError):
        temporal_row(transfer_row("source_bias_mean_fixed"))


def test_alias_cannot_replace_conflicting_saved_temporal_metadata(figure_tool):
    row = {**transfer_row("source_bias_mean_fixed"), "temporal_class": "window_endpoint"}
    with pytest.raises(ValueError):
        figure_tool.annotate_temporal_frame(pd.DataFrame([row]))
