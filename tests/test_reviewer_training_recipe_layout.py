"""Fixture geometry regression for full-precision saved training recipe text."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from test_reviewer_figures import module

TUNED_LR = 0.0022830268403520168
TUNED_WD = 0.007082535831697058
SEEDS = (20261010, 20261011, 20261012)


def training_page():
    """Synthetic histories, with exact long LR/WD and weights from failure003."""
    recipes = {
        "default": {
            "loss": "huber", "lr": 0.001, "weight_decay": 1e-5,
            "accel_signal_weight": 1.0, "derivative_weight": 0.02,
            "gyro_signal_weight": 2.0, "preintegration_weight": 0.005,
            "spectral_weight": 0.02,
        },
        "tuned": {
            "loss": "huber", "lr": TUNED_LR, "weight_decay": TUNED_WD,
            "accel_signal_weight": 1.0, "gyro_signal_weight": 1.0,
        },
    }
    records = []
    for variant, lengths, window in (
        ("default", (80, 71, 64), 80),
        ("tuned", (80, 58, 47), 320),
    ):
        for index, (seed, length) in enumerate(zip(SEEDS, lengths, strict=True)):
            for epoch in range(1, length + 1):
                records.append({
                    "cell": "alimu_only/cnn_bilstm", "job_id": f"{variant}_{seed}",
                    "protocol_id": "alimu_only", "model_family": "cnn_bilstm",
                    "task": "paired", "variant": variant, "seed": seed, "epoch": epoch,
                    "train_loss": (2 + index / 10) / (epoch + 1),
                    "val_loss": (2.2 + index / 10) / (epoch + 1),
                    "criterion": (1.5 + index / 10) / (epoch + 1),
                    "criterion_name": "val_score",
                    "loss_recipe": json.dumps(recipes[variant], sort_keys=True),
                    "window_size": np.int64(window),
                })
    return {
        "id": "training_FIXTURE_long_precision_recipe",
        "kind": "training", "title": "FIXTURE: six saved histories and exact recipe text",
        "data": pd.DataFrame(records),
    }, recipes


@pytest.fixture
def drawn_training():
    tool = module()
    tool.apply_paper_style()
    tool.matplotlib.rcParams.update(
        {"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5}
    )
    page, recipes = training_page()
    original = copy.deepcopy(page["data"])
    figure = tool.draw_page(page, tool.default_config())
    figure.canvas.draw()
    try:
        yield tool, page, recipes, original, figure
    finally:
        tool.plt.close(figure)


def test_recipe_window_sizes_are_plain_integers_and_parameters_remain_full_precision(
    drawn_training,
):
    _tool, _page, recipes, _original, figure = drawn_training
    recipe = figure.axes[5].texts[0]
    text = recipe.get_text()
    assert "np.int64" not in text
    assert "DEFAULT · T=[80]" in text and "TUNED · T=[320]" in text
    assert "Configured losses differ in scale." in text
    assert "loss: huber" in text
    assert f"lr0={TUNED_LR}" in text and f"WD={TUNED_WD}" in text
    assert "lr0=0.001" in text and "WD=1e-05" in text
    for configured in recipes.values():
        for name, value in configured.items():
            if name.endswith("_weight"):
                assert name.removesuffix("_weight") + "=" + str(value) in text
    assert recipe.get_fontsize() >= 6.0


def test_recipe_annotation_fits_its_reserved_panel_and_canvas(drawn_training):
    _tool, _page, _recipes, _original, figure = drawn_training
    renderer = figure.canvas.get_renderer()
    panel = figure.axes[5].get_window_extent(renderer)
    text = figure.axes[5].texts[0].get_window_extent(renderer)
    assert figure.get_size_inches()[0] == pytest.approx(7.2)
    assert text.x0 >= panel.x0 - 0.5 and text.x1 <= panel.x1 + 0.5, (
        "Recipe annotation exceeds reserved panel width", tuple(text.extents), tuple(panel.extents)
    )
    assert text.y0 >= panel.y0 - 0.5 and text.y1 <= panel.y1 + 0.5, (
        "Recipe annotation exceeds reserved panel height", tuple(text.extents), tuple(panel.extents)
    )
    assert text.x0 >= 0 and text.y0 >= 0
    assert text.x1 <= figure.bbox.width and text.y1 <= figure.bbox.height


def test_all_visible_training_text_remains_inside_canvas(drawn_training):
    tool, _page, _recipes, _original, figure = drawn_training
    renderer = figure.canvas.get_renderer()
    outside_ticks = set()
    for axes in figure.axes:
        for axis in (axes.xaxis, axes.yaxis):
            lower, upper = sorted(axis.get_view_interval())
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if tick.get_loc() < lower or tick.get_loc() > upper:
                    outside_ticks.update((id(tick.label1), id(tick.label2)))
    outside = []
    for text in figure.findobj(tool.matplotlib.text.Text):
        if not text.get_visible() or not text.get_text() or id(text) in outside_ticks:
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > figure.bbox.width or box.y1 > figure.bbox.height:
            outside.append((text.get_text(), tuple(box.extents)))
    assert not outside


def test_six_histories_common_criterion_and_exact_source_recipe_are_unchanged(drawn_training):
    _tool, page, _recipes, original, figure = drawn_training
    pd.testing.assert_frame_equal(page["data"], original)
    assert original.job_id.nunique() == 6
    for variant, axes_indices in (("default", (0, 3)), ("tuned", (1, 4))):
        part = original[original.variant.eq(variant)]
        for value, axes_index in zip(("train_loss", "val_loss"), axes_indices, strict=True):
            lines = figure.axes[axes_index].lines
            assert len(lines) == 3
            assert [line.get_label() for line in lines] == list(map(str, SEEDS))
            for line, seed in zip(lines, SEEDS, strict=True):
                source = part[part.seed.eq(seed)].sort_values("epoch")
                np.testing.assert_array_equal(line.get_xdata(), source.epoch)
                np.testing.assert_array_equal(line.get_ydata(), source[value])
    criterion = figure.axes[2]
    assert criterion.get_title() == "val_score"
    assert criterion.get_ylabel() == "Shared validation criterion"
    assert len(criterion.lines) == 6
    for line, (variant, seed) in zip(
        criterion.lines, [(variant, seed) for variant in ("default", "tuned") for seed in SEEDS],
        strict=True,
    ):
        source = original[original.variant.eq(variant) & original.seed.eq(seed)].sort_values("epoch")
        np.testing.assert_array_equal(line.get_xdata(), source.epoch)
        np.testing.assert_array_equal(line.get_ydata(), source.criterion)
        assert line.get_linestyle() == ("-" if variant == "default" else "--")
    assert criterion.get_legend() is None
    assert len(figure.legends) == 1
    legend = figure.legends[0]
    assert [text.get_text() for text in legend.get_texts()] == [
        f"{seed} {variant}" for variant in ("D", "T") for seed in SEEDS
    ]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    assert all(not legend_box.overlaps(axes.get_tightbbox(renderer)) for axes in figure.axes)


@pytest.mark.parametrize("field", ["criterion_name", "loss_recipe"])
def test_training_layout_does_not_relax_mixed_recipe_or_criterion_guard(field):
    tool = module()
    page, _recipes = training_page()
    page["data"].loc[0, field] = "different" if field == "criterion_name" else "{}"
    try:
        with pytest.raises(ValueError, match=r"mix criteria|mixes different configured loss recipes"):
            tool.draw_page(page, tool.default_config())
    finally:
        tool.plt.close("all")
