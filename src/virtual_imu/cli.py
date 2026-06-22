"""Typer CLI and callable pipeline functions."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer
from loguru import logger

from virtual_imu.config import load_config
from virtual_imu.eval.baselines import evaluate_baselines as evaluate_baseline_set
from virtual_imu.eval.reports import evaluate_predictions
from virtual_imu.inference.virtual_imu_generator import VirtualIMUGenerator
from virtual_imu.io.download import download_smartpnt
from virtual_imu.io.parquet_store import write_parquet
from virtual_imu.io.smartpnt_reader import SmartPNTReader, scan_dataset
from virtual_imu.logging import setup_logging
from virtual_imu.preprocessing.split import split_by_fraction
from virtual_imu.preprocessing.sync import align_low_high
from virtual_imu.preprocessing.windowing import window_long_dataframe
from virtual_imu.sim.paired_dataset import generate_paired_dataset
from virtual_imu.training.trainer import train_model
from virtual_imu.viz.dashboard import plot_all_results

app = typer.Typer(help="Virtual high-grade IMU research toolkit.")


def run_artifact_paths(run_dir: str | Path) -> dict[str, Path]:
    """Return the standard artifact layout for one experiment run."""
    root = Path(run_dir)
    return {
        "prediction": root / "predictions" / "virtual_imu.parquet",
        "eval": root / "eval" / "eval_metrics.json",
        "baseline": root / "baselines" / "baseline_metrics.json",
        "baseline_predictions": root / "baselines" / "predictions",
        "baseline_figures": root / "baselines" / "figures",
        "figures": root / "figures",
        "report": root / "report.md",
    }


def publish_latest(run_dir: str | Path, outputs_root: str | Path = "outputs") -> Path:
    """Publish lightweight pointers and copies for the latest completed run."""
    run_path = Path(run_dir)
    out_root = Path(outputs_root)
    latest = out_root / "latest"
    paths = run_artifact_paths(run_path)
    latest.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "latest_run.txt").write_text(str(run_path), encoding="utf-8")

    copy_pairs = [
        (paths["prediction"], latest / "virtual_imu.parquet"),
        (paths["eval"], latest / "eval_metrics.json"),
        (paths["eval"].with_suffix(".csv"), latest / "eval_metrics.csv"),
        (paths["eval"].with_suffix(".md"), latest / "eval_metrics.md"),
        (paths["baseline"], latest / "baseline_metrics.json"),
        (paths["baseline"].with_suffix(".csv"), latest / "baseline_metrics.csv"),
        (paths["baseline"].with_suffix(".md"), latest / "baseline_metrics.md"),
        (paths["report"], latest / "report.md"),
    ]
    for src, dst in copy_pairs:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    if paths["figures"].exists():
        shutil.copytree(paths["figures"], latest / "figures", dirs_exist_ok=True)
    if paths["baseline_figures"].exists():
        shutil.copytree(paths["baseline_figures"], latest / "baseline_figures", dirs_exist_ok=True)
    return latest


def make_sim_data(config: str | Path) -> dict[str, Path]:
    """Generate paired simulated IMU data from config."""
    setup_logging()
    cfg = load_config(config)
    return generate_paired_dataset(cfg, config_path=config)


def inspect_dataset(root: str | Path, out: str | Path | None = None, preview_limit: int = 5) -> dict[str, Any]:
    """Inspect a raw dataset directory and optionally write Markdown."""
    setup_logging()
    report = scan_dataset(root, preview_limit=preview_limit)
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_inspection_markdown(report), encoding="utf-8")
        logger.info("Wrote inspection report to {}", out_path)
    else:
        typer.echo(_inspection_markdown(report))
    return report


def prepare_smartpnt(config: str | Path) -> Path | None:
    """Prepare SmartPNT-MSF after config-driven inspection and mapping."""
    setup_logging()
    cfg = load_config(config)
    out_dir = Path(cfg.get("output_dir", "data/processed/smartpnt"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(cfg.get("report_path", "outputs/smartpnt_preparation_report.md"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    reader = SmartPNTReader(cfg)
    inspection = reader.inspect()
    if not inspection.get("exists", False):
        report_path.write_text(_preparation_missing_report(cfg, inspection), encoding="utf-8")
        logger.warning("SmartPNT raw data missing. Wrote guidance to {}", report_path)
        return None
    if not cfg.get("low_imu_source") or not cfg.get("high_imu_source"):
        report_path.write_text(_preparation_missing_report(cfg, inspection), encoding="utf-8")
        logger.warning("SmartPNT sources not configured. Wrote guidance to {}", report_path)
        return None
    low, high = reader.read_pair()
    sync_cfg = cfg.get("sync", {})
    aligned, align_report = align_low_high(
        low,
        high,
        target_rate_hz=float(cfg.get("target_rate_hz", 100.0)),
        method=str(sync_cfg.get("method", "linear")),
        time_offset_s=float(cfg.get("time_offset_s", 0.0)),
        estimate_delay=bool(sync_cfg.get("estimate_delay", True)),
        max_lag_s=float(sync_cfg.get("max_lag_s", 0.5)),
    )
    aligned_path = write_parquet(aligned, out_dir / "aligned.parquet")
    splits = split_by_fraction(aligned, **cfg.get("split", {}))
    win_cfg = cfg.get("windowing", {})
    rate = float(cfg.get("target_rate_hz", 100.0))
    window_size = max(1, int(round(float(win_cfg.get("input_window_s", 2.0)) * rate)))
    stride = max(1, int(round(float(win_cfg.get("stride_s", 0.5)) * rate)))
    for name, frame in splits.items():
        windows = window_long_dataframe(frame, window_size, stride, target=str(win_cfg.get("target", "full_sequence")))
        write_parquet(windows, out_dir / f"windows_{name}.parquet")
    report_path.write_text(_preparation_report(aligned_path, align_report, inspection), encoding="utf-8")
    logger.info("Prepared SmartPNT data under {}", out_dir)
    return aligned_path


def train(config: str | Path) -> Path:
    """Train a virtual IMU model from config."""
    setup_logging()
    return train_model(load_config(config), config_path=config)


def infer(checkpoint: str | Path, input_path: str | Path, out: str | Path, device: str = "auto") -> Path:
    """Generate virtual high-grade IMU predictions."""
    setup_logging()
    return VirtualIMUGenerator(checkpoint, device=device).run(input_path, out)


def evaluate(pred: str | Path, out: str | Path, min_allan_duration_s: float = 20.0) -> dict[str, Any]:
    """Evaluate a prediction Parquet file."""
    setup_logging()
    metrics = evaluate_predictions(pred, out, min_allan_duration_s=min_allan_duration_s)
    logger.info("Wrote evaluation metrics to {}", out)
    return metrics


def evaluate_baselines(
    input_path: str | Path,
    out: str | Path,
    train_path: str | Path | None = None,
    predictions_dir: str | Path | None = None,
    min_allan_duration_s: float = 20.0,
    moving_average_window: int = 5,
    butterworth_cutoff_hz: float = 12.0,
    savgol_window: int = 9,
    fir_taps: int = 9,
    neural_epochs: int = 4,
    neural_window_size: int = 128,
    neural_stride: int = 64,
) -> dict[str, Any]:
    """Evaluate the three-tier baseline suite."""
    setup_logging()
    metrics = evaluate_baseline_set(
        input_path,
        out,
        train_path=train_path,
        predictions_dir=predictions_dir,
        min_allan_duration_s=min_allan_duration_s,
        moving_average_window=moving_average_window,
        butterworth_cutoff_hz=butterworth_cutoff_hz,
        savgol_window=savgol_window,
        fir_taps=fir_taps,
        neural_epochs=neural_epochs,
        neural_window_size=neural_window_size,
        neural_stride=neural_stride,
    )
    logger.info("Wrote baseline metrics to {}", out)
    return metrics


def plot_results(eval_path: str | Path, pred: str | Path, out: str | Path) -> Path:
    """Generate paper-style figures."""
    setup_logging()
    index = plot_all_results(eval_path, pred, out)
    logger.info("Wrote figure index to {}", index)
    return index


def make_report(
    run: str | Path,
    out: str | Path,
    metrics: str | Path | None = None,
    figures: str | Path | None = None,
    baselines: str | Path | None = None,
) -> Path:
    """Generate a Markdown experiment report."""
    setup_logging()
    run_path = Path(run)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    paths = run_artifact_paths(run_path)
    metrics_path = Path(metrics) if metrics else paths["eval"]
    if not metrics_path.exists():
        metrics_path = Path("outputs/eval_metrics.json")
    figure_index = Path(figures) / "figure_index.md" if figures else paths["figures"] / "figure_index.md"
    if not figure_index.exists():
        figure_index = Path("outputs/figures/figure_index.md")
    baseline_path = Path(baselines) if baselines else paths["baseline"]
    if not baseline_path.exists():
        baseline_path = Path("outputs/baseline_metrics.json")
    baseline_figure_index = baseline_path.parent / "figures" / "baseline_figure_index.md"
    if not baseline_figure_index.exists():
        baseline_figure_index = Path("outputs/latest/baseline_figures/baseline_figure_index.md")
    lines = ["# Virtual IMU Experiment Report", ""]
    lines.append("## Data Source")
    lines.append("")
    lines.append("Simulation data is used by default unless a SmartPNT preparation report is present.")
    sim_manifest = Path("data/processed/sim/simulation_manifest.json")
    if sim_manifest.exists():
        lines.append(f"- Simulation manifest: `{sim_manifest}`")
    smartpnt_report = Path("outputs/smartpnt_preparation_report.md")
    if smartpnt_report.exists():
        lines.append(f"- SmartPNT preparation report: `{smartpnt_report}`")
    lines.extend(["", "## Processing Configuration", ""])
    cfg_path = run_path / "config.yaml"
    if cfg_path.exists():
        lines.append(f"- Run config: `{cfg_path}`")
    else:
        lines.append(f"- Run directory not found or no config snapshot: `{run_path}`")
    lines.extend(["", "## Model Configuration", ""])
    if cfg_path.exists():
        cfg = load_config(cfg_path)
        lines.append(f"- Model type: `{cfg.get('model', {}).get('type', 'unknown')}`")
        lines.append(f"- Seed: `{cfg.get('seed', 'unknown')}`")
    lines.extend(["", "## Metrics Table", ""])
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        sig = metrics.get("signal", {})
        for name in ("low", "virtual"):
            vals = sig.get(name, {})
            lines.append(
                f"- {name}: gyro norm RMSE={vals.get('gyro_norm_rmse', float('nan')):.6g}, "
                f"accel norm RMSE={vals.get('accel_norm_rmse', float('nan')):.6g}"
            )
    else:
        lines.append("- Metrics file not found: `outputs/eval_metrics.json`")
    lines.extend(["", "## Baseline Comparison", ""])
    baseline_md = baseline_path.with_suffix(".md")
    if baseline_md.exists():
        lines.append(baseline_md.read_text(encoding="utf-8"))
    elif baseline_path.exists():
        lines.append(f"- Baseline metrics: `{baseline_path}`")
    else:
        lines.append("- Baseline metrics not found. Run `scripts/evaluate_baselines.py` or `scripts/run_experiment.py`.")
    if baseline_figure_index.exists():
        lines.extend(["", "### Baseline Figures", ""])
        lines.append(baseline_figure_index.read_text(encoding="utf-8"))
    lines.extend(["", "## Main Figures", ""])
    if figure_index.exists():
        lines.append(figure_index.read_text(encoding="utf-8"))
    else:
        lines.append("- Figure index not found.")
    lines.extend(
        [
            "",
            "## Failure Cases",
            "",
            "- Review time windows where virtual-vs-high residuals remain larger than low-vs-high residuals.",
            "- Inspect high-dynamic vibration periods for saturation or clipping.",
            "",
            "## Limitations",
            "",
            "- High-grade IMU integration is a proxy navigation reference when no external ground truth is available.",
            "- The simulation covers common low-cost IMU errors but is not a substitute for real SmartPNT validation.",
            "",
            "## Next Steps",
            "",
            "- Inspect SmartPNT-MSF raw files and update `configs/smartpnt_example.yaml` with true sources/columns.",
            "- Add uncertainty heads and domain randomization before deployment.",
            "- Compare TCN, GRU, Transformer, U-Net, affine, and filtering baselines on matched splits.",
            "",
            "## Caption Drafts",
            "",
            "- See the run-local `figures/figure_index.md` for per-figure caption drafts.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote report to {}", out_path)
    return out_path


def run_experiment(
    train_config: str | Path,
    test_input: str | Path,
    baseline_train: str | Path | None = None,
    device: str = "auto",
    publish: bool = True,
    baseline_neural_epochs: int = 4,
    baseline_neural_window_size: int = 128,
    baseline_neural_stride: int = 64,
) -> Path:
    """Train, infer, evaluate, compare baselines, plot, and report into one run directory."""
    setup_logging()
    run_dir = train(train_config)
    paths = run_artifact_paths(run_dir)
    infer(run_dir / "best.ckpt", test_input, paths["prediction"], device=device)
    evaluate(paths["prediction"], paths["eval"])
    cfg = load_config(train_config)
    baseline_train_path = baseline_train or cfg.get("data", {}).get("train_path")
    evaluate_baselines(
        test_input,
        paths["baseline"],
        train_path=baseline_train_path,
        predictions_dir=paths["baseline_predictions"],
        neural_epochs=baseline_neural_epochs,
        neural_window_size=baseline_neural_window_size,
        neural_stride=baseline_neural_stride,
    )
    plot_results(paths["eval"], paths["prediction"], paths["figures"])
    make_report(
        run_dir,
        paths["report"],
        metrics=paths["eval"],
        figures=paths["figures"],
        baselines=paths["baseline"],
    )
    if publish:
        latest = publish_latest(run_dir)
        logger.info("Published latest-run snapshot to {}", latest)
    logger.info("Completed experiment run at {}", run_dir)
    return run_dir


def run_smoke_test() -> None:
    """Run a complete small simulation/train/infer/evaluate/plot/report/test loop."""
    setup_logging()
    smoke_sim_cfg = {
        "seed": 11,
        "output_dir": "data/processed/sim",
        "rate_hz": 50.0,
        "splits": {
            "train": {"duration_s": 12.0, "sequence_id": "smoke_train"},
            "val": {"duration_s": 6.0, "sequence_id": "smoke_val"},
            "test": {"duration_s": 6.0, "sequence_id": "smoke_test"},
        },
        "motion": {"vibration_hz": 9.0, "max_turn_rate_rad_s": 0.8},
        "imu_error": {
            "gyro_bias_rad_s": [0.01, -0.006, 0.004],
            "accel_bias_m_s2": [0.05, -0.03, 0.02],
            "gyro_noise_std": 0.006,
            "accel_noise_std": 0.06,
            "bias_random_walk_std": 0.0005,
            "ar1_alpha": 0.9,
            "ar1_std": 0.003,
            "time_delay_s": 0.01,
            "quantization": {"gyro": 0.0003, "accel": 0.004},
            "clipping": {"gyro": 6.0, "accel": 80.0},
        },
    }
    generate_paired_dataset(smoke_sim_cfg, config_path="smoke-inline")
    smoke_train_cfg = {
        "seed": 11,
        "experiment_name": "smoke_tcn",
        "data": {
            "train_path": "data/processed/sim/train.parquet",
            "val_path": "data/processed/sim/val.parquet",
            "window_size": 64,
            "stride": 16,
            "batch_size": 8,
            "num_workers": 0,
        },
        "model": {"type": "tcn", "hidden_channels": 16, "levels": 2, "kernel_size": 3, "dropout": 0.0},
        "training": {
            "epochs": 2,
            "lr": 0.002,
            "weight_decay": 0.0,
            "loss": "huber",
            "derivative_weight": 0.01,
            "spectral_weight": 0.0,
            "allan_weight": 0.0,
            "preintegration_weight": 0.0,
            "grad_clip_norm": 1.0,
            "early_stopping_patience": 3,
            "device": "cpu",
        },
        "output": {"runs_dir": "runs"},
    }
    run_dir = train_model(smoke_train_cfg, config_path="smoke-inline")
    paths = run_artifact_paths(run_dir)
    infer(run_dir / "best.ckpt", "data/processed/sim/test.parquet", paths["prediction"], device="cpu")
    evaluate(paths["prediction"], paths["eval"], min_allan_duration_s=20.0)
    evaluate_baselines(
        "data/processed/sim/test.parquet",
        paths["baseline"],
        train_path="data/processed/sim/train.parquet",
        predictions_dir=paths["baseline_predictions"],
        min_allan_duration_s=20.0,
        neural_epochs=1,
        neural_window_size=64,
        neural_stride=32,
    )
    plot_results(paths["eval"], paths["prediction"], paths["figures"])
    make_report(run_dir, paths["report"], metrics=paths["eval"], figures=paths["figures"], baselines=paths["baseline"])
    publish_latest(run_dir)
    logger.info("Running pytest")
    subprocess.run([sys.executable, "-m", "pytest"], check=True)
    logger.info("Smoke test completed")


def _inspection_markdown(report: dict[str, Any]) -> str:
    lines = ["# Dataset Inspection", ""]
    if not report.get("exists", False):
        return "\n".join([*lines, report.get("message", "Dataset not found."), ""]) + "\n"
    lines.extend(
        [
            f"- Root: `{report.get('root')}`",
            f"- File count: {report.get('file_count')}",
            f"- Total size bytes: {report.get('total_size_bytes')}",
            f"- Max file size bytes: {report.get('max_size_bytes')}",
            "",
            "## Suffix Counts",
            "",
        ]
    )
    for suffix, count in sorted(report.get("suffix_counts", {}).items()):
        lines.append(f"- `{suffix}`: {count}")
    lines.extend(["", "## Directory Tree Preview", ""])
    for item in report.get("tree", [])[:80]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## File Previews", ""])
    for preview in report.get("previews", []):
        lines.append(f"### `{preview.get('path')}`")
        lines.append("")
        lines.append(f"- Kind: {preview.get('kind')}")
        if preview.get("columns"):
            lines.append(f"- Columns: {preview.get('columns')}")
        if preview.get("keys"):
            lines.append(f"- Keys: {preview.get('keys')}")
        if preview.get("note"):
            lines.append(f"- Note: {preview.get('note')}")
        if preview.get("error"):
            lines.append(f"- Error: {preview.get('error')}")
        lines.append("")
    return "\n".join(lines)


def _preparation_missing_report(cfg: dict[str, Any], inspection: dict[str, Any]) -> str:
    return (
        "# SmartPNT Preparation Report\n\n"
        "SmartPNT-MSF data is not ready for parsing yet.\n\n"
        f"- Raw root: `{cfg.get('raw_root', 'data/raw/smartpnt')}`\n"
        f"- Inspection status: `{inspection.get('message', 'sources not configured')}`\n\n"
        "Next steps:\n\n"
        "1. Download or manually place SmartPNT-MSF files under `data/raw/smartpnt`.\n"
        "2. Run `python scripts/inspect_dataset.py --root data/raw/smartpnt --out outputs/smartpnt_inspection.md`.\n"
        "3. Set `low_imu_source`, `high_imu_source`, and column names in `configs/smartpnt_example.yaml`.\n"
    )


def _preparation_report(aligned_path: Path, align_report: dict[str, Any], inspection: dict[str, Any]) -> str:
    lines = [
        "# SmartPNT Preparation Report",
        "",
        f"- Aligned output: `{aligned_path}`",
        f"- File count inspected: {inspection.get('file_count')}",
        "",
        "## Alignment",
        "",
    ]
    for key, value in align_report.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


@app.command("make-sim-data")
def make_sim_data_cmd(config: Path = typer.Option(..., "--config")) -> None:
    make_sim_data(config)


@app.command("download-smartpnt")
def download_smartpnt_cmd(
    repo_url: str = typer.Option("https://gitee.com/lv-jiarui/SmartPNT-MSF-Datasets", "--repo-url"),
    out: Path = typer.Option(Path("data/raw/smartpnt"), "--out"),
    manifest: Path = typer.Option(Path("configs/smartpnt_manifest.yaml"), "--manifest"),
    sha256: bool = typer.Option(False, "--sha256"),
) -> None:
    setup_logging()
    download_smartpnt(repo_url, out, manifest_config=manifest, include_sha256=sha256)


@app.command("inspect-dataset")
def inspect_dataset_cmd(
    root: Path = typer.Option(Path("data/raw/smartpnt"), "--root"),
    out: Path | None = typer.Option(None, "--out"),
    preview_limit: int = typer.Option(5, "--preview-limit"),
) -> None:
    inspect_dataset(root, out=out, preview_limit=preview_limit)


@app.command("prepare-smartpnt")
def prepare_smartpnt_cmd(config: Path = typer.Option(..., "--config")) -> None:
    prepare_smartpnt(config)


@app.command("train")
def train_cmd(config: Path = typer.Option(..., "--config")) -> None:
    run_dir = train(config)
    typer.echo(run_dir)


@app.command("infer")
def infer_cmd(
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    input_path: Path = typer.Option(..., "--input"),
    out: Path = typer.Option(..., "--out"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    infer(checkpoint, input_path, out, device=device)


@app.command("evaluate")
def evaluate_cmd(
    pred: Path = typer.Option(..., "--pred"),
    out: Path = typer.Option(..., "--out"),
    min_allan_duration_s: float = typer.Option(20.0, "--min-allan-duration-s"),
) -> None:
    evaluate(pred, out, min_allan_duration_s=min_allan_duration_s)


@app.command("evaluate-baselines")
def evaluate_baselines_cmd(
    input_path: Path = typer.Option(..., "--input"),
    out: Path = typer.Option(..., "--out"),
    train_path: Path | None = typer.Option(None, "--train"),
    predictions_dir: Path | None = typer.Option(None, "--predictions-dir"),
    min_allan_duration_s: float = typer.Option(20.0, "--min-allan-duration-s"),
    moving_average_window: int = typer.Option(5, "--moving-average-window"),
    butterworth_cutoff_hz: float = typer.Option(12.0, "--butterworth-cutoff-hz"),
    savgol_window: int = typer.Option(9, "--savgol-window"),
    fir_taps: int = typer.Option(9, "--fir-taps"),
    neural_epochs: int = typer.Option(4, "--neural-epochs"),
    neural_window_size: int = typer.Option(128, "--neural-window-size"),
    neural_stride: int = typer.Option(64, "--neural-stride"),
) -> None:
    evaluate_baselines(
        input_path,
        out,
        train_path=train_path,
        predictions_dir=predictions_dir,
        min_allan_duration_s=min_allan_duration_s,
        moving_average_window=moving_average_window,
        butterworth_cutoff_hz=butterworth_cutoff_hz,
        savgol_window=savgol_window,
        fir_taps=fir_taps,
        neural_epochs=neural_epochs,
        neural_window_size=neural_window_size,
        neural_stride=neural_stride,
    )


@app.command("plot-results")
def plot_results_cmd(
    eval_path: Path = typer.Option(..., "--eval"),
    pred: Path = typer.Option(..., "--pred"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    plot_results(eval_path, pred, out)


@app.command("make-report")
def make_report_cmd(
    run: Path = typer.Option(..., "--run"),
    out: Path = typer.Option(..., "--out"),
    metrics: Path | None = typer.Option(None, "--metrics"),
    figures: Path | None = typer.Option(None, "--figures"),
    baselines: Path | None = typer.Option(None, "--baselines"),
) -> None:
    make_report(run, out, metrics=metrics, figures=figures, baselines=baselines)


@app.command("run-experiment")
def run_experiment_cmd(
    train_config: Path = typer.Option(..., "--train-config"),
    test_input: Path = typer.Option(..., "--test"),
    baseline_train: Path | None = typer.Option(None, "--baseline-train"),
    device: str = typer.Option("auto", "--device"),
    publish: bool = typer.Option(True, "--publish/--no-publish"),
    baseline_neural_epochs: int = typer.Option(4, "--baseline-neural-epochs"),
    baseline_neural_window_size: int = typer.Option(128, "--baseline-neural-window-size"),
    baseline_neural_stride: int = typer.Option(64, "--baseline-neural-stride"),
) -> None:
    run_dir = run_experiment(
        train_config,
        test_input,
        baseline_train=baseline_train,
        device=device,
        publish=publish,
        baseline_neural_epochs=baseline_neural_epochs,
        baseline_neural_window_size=baseline_neural_window_size,
        baseline_neural_stride=baseline_neural_stride,
    )
    typer.echo(run_dir)


@app.command("run-smoke-test")
def run_smoke_test_cmd() -> None:
    run_smoke_test()


if __name__ == "__main__":
    app()
