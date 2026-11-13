"""Reporting checks use physical metric fixtures, never training jobs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from virtual_imu.viz import plots_matched as reporting


def _metrics():
    rows = []
    for pair, scale in (("A_to_VN100", 1), ("B_to_VN100", 9), ("__all_pairs__", 4)):
        for component, unit, factor in (("gyro", "rad/s", 1), ("accel", "m/s^2", 10)):
            for seed in (11, 12, 13):
                for method in ("ddpm", "raw", "affine"):
                    rmse = factor * (scale + seed - 11 if method == "ddpm" else scale + 5)
                    rows.append(
                        dict(
                            pair_id=pair,
                            seed=seed,
                            method=method,
                            component=component,
                            unit=unit,
                            samples=100,
                            rmse=rmse,
                        )
                    )
    return pd.DataFrame(rows)


def test_summary_uses_physical_pooled_rows_and_sample_seed_sd():
    summary = reporting._summarize_metrics(_metrics())
    row = summary.query(
        "pair_id == '__all_pairs__' and method == 'ddpm' and component == 'gyro'"
    ).iloc[0]
    assert row.rmse_mean == 5  # Not mean(A, B), which would be 6.
    assert row.rmse_sd == 1  # ddof=1, not population SD.
    assert row.vector_rmse_mean == pytest.approx(np.sqrt(3) * 5)
    assert row.vector_rmse_sd == pytest.approx(np.sqrt(3))
    assert row.n == 3
    assert row.samples == 100  # Repeated training seeds do not add physical samples.
    baseline = summary[summary.method.isin(["raw", "affine"])]
    assert baseline.rmse_sd.isna().all()
    assert baseline.n.eq(1).all()
    accel = summary.query(
        "pair_id == '__all_pairs__' and method == 'ddpm' and component == 'accel'"
    ).iloc[0]
    assert accel.rmse_mean == 50
    assert accel.unit == "m/s^2"


def test_summary_does_not_invent_pooled_metrics():
    metrics = _metrics().query("pair_id != '__all_pairs__'")
    assert "__all_pairs__" not in set(reporting._summarize_metrics(metrics).pair_id)


def test_missing_inputs_write_incomplete_report_without_numeric_results(tmp_path, caplog):
    reporting.render_matched_report(tmp_path, {})
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "输入不完整" in report
    assert "metrics_per_seed.csv" in report
    assert "无可用" in report
    assert "实验已完成" not in report
    assert "outputs/latest" not in report
    assert (tmp_path / "figures/figure_index.md").exists()
    assert not list((tmp_path / "figures").glob("*.png"))
    assert "SKIPS" in caplog.text


def test_report_retains_negative_strict_deltas_and_reports_failed_initialization(
    tmp_path, monkeypatch
):
    _metrics().to_csv(tmp_path / "metrics_per_seed.csv", index=False)
    pd.DataFrame(
        [
            dict(
                pair_id="A_to_VN100",
                seed=11,
                component="gyro",
                ddpm_rmse=0.8,
                no_diffusion_rmse=1.0,
                ddpm_change_pct=-20.0,
                initial_backbone_matches=True,
            ),
            dict(
                pair_id="A_to_VN100",
                seed=12,
                component="gyro",
                ddpm_rmse=1.1,
                no_diffusion_rmse=1.0,
                ddpm_change_pct=10.0,
                initial_backbone_matches=False,
            ),
        ]
    ).to_csv(tmp_path / "matched_seed_deltas.csv", index=False)
    observed = []

    def capture(fig, directory, stem):
        if "delta" in stem:
            for ax in fig.axes:
                for collection in ax.collections:
                    observed.extend(collection.get_offsets()[:, 1])
        reporting.plt.close(fig)
        return []

    monkeypatch.setattr(reporting, "save_figure", capture)
    reporting.render_matched_report(tmp_path, dict(seeds=[11, 12, 13]))
    assert min(observed) == -20.0
    assert max(observed) == 10.0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "-20" in report
    assert "初始化不一致" in report
    assert "正值表示 DDPM 更差" in report
    assert "独立录制" in report
    assert "VN100" in report


def test_real_exports_keep_units_separate_and_use_relative_links(tmp_path):
    metrics = _metrics().query("pair_id == '__all_pairs__'")
    metrics.to_csv(tmp_path / "metrics_per_seed.csv", index=False)
    reporting._summarize_metrics(metrics).to_csv(tmp_path / "summary.csv", index=False)
    (tmp_path / "provenance.json").write_text(json.dumps({"git_commit": "abc"}))
    reporting.render_matched_report(tmp_path, dict(seeds=[11, 12, 13]))
    stem = tmp_path / "figures/rmse_pooled"
    for extension in ("png", "svg", "pdf"):
        assert stem.with_suffix("." + extension).stat().st_size > 1000
    svg = stem.with_suffix(".svg").read_text(encoding="utf-8")
    assert "rad/s" in svg
    assert "m/s" in svg
    index = (tmp_path / "figures/figure_index.md").read_text(encoding="utf-8")
    assert "rmse_pooled.svg" in index
    assert str(tmp_path) not in index
    assert "sample" in index.lower() or "样本标准差" in index


def test_malformed_csv_is_not_treated_as_finished(tmp_path):
    (tmp_path / "metrics_per_seed.csv").write_text("pair_id,method\nA,ddpm\n")
    reporting.render_matched_report(tmp_path, {})
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "输入不完整" in report
    assert "缺少字段" in report
    assert not list((tmp_path / "figures").glob("*.png"))


@pytest.mark.parametrize("field,value", [("samples", 99), ("rmse", float("nan"))])
def test_summary_rejects_invalid_seed_support_or_metrics(field, value):
    metrics = _metrics()
    metrics.loc[0, field] = value
    with pytest.raises(ValueError):
        reporting._summarize_metrics(metrics)


def test_small_dynamic_run_with_flattened_trial_fields_and_optional_histories(tmp_path):
    metrics = _metrics().query("pair_id != 'B_to_VN100' and seed != 13").copy()
    metrics = metrics[metrics.method.eq("ddpm")]
    direct = metrics.assign(method="no_diffusion", rmse=metrics.rmse * 1.1)
    strict = metrics.assign(method="no_diffusion_matched", rmse=metrics.rmse * 1.2)
    metrics = pd.concat([metrics, direct, strict], ignore_index=True)
    metrics.to_csv(tmp_path / "metrics_per_seed.csv", index=False)
    reporting._summarize_metrics(metrics).to_csv(tmp_path / "summary.csv", index=False)
    pd.DataFrame(
        [
            dict(
                pair_id="A_to_VN100",
                seed=seed,
                component=component,
                ddpm_rmse=1.0,
                no_diffusion_rmse=2.0,
                ddpm_change_pct=-50.0,
                initial_backbone_matches=True,
            )
            for seed in (11, 12)
            for component in ("gyro", "accel")
        ]
    ).to_csv(tmp_path / "matched_seed_deltas.csv", index=False)
    pd.json_normalize(
        [
            dict(
                pair_id="A_to_VN100",
                method=method,
                seed=11,
                candidate=0,
                validation_score=0.9,
                config=dict(model=dict(d_model=8)),
            )
            for method in ("ddpm", "no_diffusion")
        ]
    ).to_csv(tmp_path / "tuning_trials.csv", index=False)
    pd.DataFrame(
        [
            dict(
                pair_id="A_to_VN100",
                method=method,
                seed=11,
                batch_size=1,
                window_points=16,
                draws=draws,
                reverse_steps=steps,
                mean_batch_ms=ms,
                p50_batch_ms=ms,
                mean_window_ms=ms,
                parameters=100,
            )
            for method, draws, steps, ms in (("ddpm", 1, 3, 10.0), ("no_diffusion", 1, 0, 0.1))
        ]
    ).to_csv(tmp_path / "latency.csv", index=False)
    (tmp_path / "selections.lock.json").write_text(
        json.dumps(
            dict(
                test_used_for_selection=False,
                selections=[
                    dict(pair_id="A_to_VN100", method=m, candidate=0)
                    for m in ("ddpm", "no_diffusion")
                ],
            )
        )
    )
    (tmp_path / "provenance.json").write_text(json.dumps(dict(git_commit="abc")))
    history = tmp_path / "fits/A_to_VN100/ddpm/search_trial_000/history.csv"
    history.parent.mkdir(parents=True)
    pd.DataFrame(dict(epoch=[1, 2], train_loss=[1.0, 0.5], validation_score=[1.0, 0.9])).to_csv(
        history, index=False
    )
    archived = history.parent.with_name("search_trial_000_interrupted_123")
    archived.mkdir()
    (archived / "history.csv").write_text("broken\n", encoding="utf-8")
    config = dict(
        seeds=[11, 12],
        methods=["ddpm", "no_diffusion"],
        data=dict(pairs=["A_to_VN100"], window_size=16, stride=8, eval_stride=8),
    )
    reporting.render_matched_report(tmp_path, config)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "输入不完整或存在一致性问题" not in report
    assert "n=3" not in report
    assert "search_trial_000_interrupted" not in report
    assert (tmp_path / "figures/history_A_to_VN100_ddpm.png").is_file()
    assert (tmp_path / "figures/runtime_batch_1_window_16.svg").is_file()
    assert "log scale" in (tmp_path / "figures/runtime_batch_1_window_16.svg").read_text()
    assert "[11, 12]" in report
