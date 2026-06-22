# smartpnt-virtual-imu

`smartpnt-virtual-imu` is a reproducible Python research project for learning a virtual high-grade IMU from low-cost or lower-grade IMU signals. The first milestone is a complete simulation loop that runs without real data; the second milestone connects the same pipeline to SmartPNT-MSF after inspecting the downloaded files.

Public resources:

- SmartPNT-MSF paper: <https://arxiv.org/abs/2507.19079>
- SmartPNT-MSF dataset repo: <https://gitee.com/lv-jiarui/SmartPNT-MSF-Datasets>

## Quick Start

```bash
uv sync
uv run python scripts/run_smoke_test.py
```

The smoke test generates a tiny paired IMU simulation, trains a small TCN for 1-2 epochs, runs inference, evaluates metrics, exports paper-style figures, creates a report, and runs `pytest`.

## Simulation Loop

Recommended one-command experiment after generating simulation data:

```bash
uv run python scripts/make_sim_data.py --config configs/sim_baseline.yaml
uv run python scripts/run_experiment.py \
  --train-config configs/train_tcn.yaml \
  --test data/processed/sim/test.parquet \
  --baseline-train data/processed/sim/train.parquet \
  --baseline-neural-epochs 4
```

This writes all model predictions, metrics, baseline comparisons, figures, and the report under one run directory:

```text
runs/<id>/
  best.ckpt
  config.yaml
  predictions/virtual_imu.parquet
  eval/eval_metrics.json
  baselines/baseline_metrics.json
  baselines/baseline_metrics.csv
  baselines/baseline_metrics.md
  baselines/figures/
  figures/
  report.md
```

It also publishes a convenience copy under `outputs/latest/` and writes `outputs/latest_run.txt`.

Manual step-by-step commands are still supported:

```bash
uv run python scripts/make_sim_data.py --config configs/sim_baseline.yaml
uv run python scripts/train.py --config configs/train_tcn.yaml
uv run python scripts/infer.py --checkpoint runs/<id>/best.ckpt --input data/processed/sim/test.parquet --out outputs/virtual_imu.parquet
uv run python scripts/evaluate.py --pred outputs/virtual_imu.parquet --out outputs/eval_metrics.json
uv run python scripts/evaluate_baselines.py --input data/processed/sim/test.parquet --train data/processed/sim/train.parquet --out outputs/baseline_metrics.json
uv run python scripts/plot_results.py --eval outputs/eval_metrics.json --pred outputs/virtual_imu.parquet --out outputs/figures
uv run python scripts/make_report.py --run runs/<id> --out outputs/report.md
```

## Baseline Suite

`scripts/evaluate_baselines.py` now produces a three-layer comparison table and Nature-style baseline figures:

- L1 classical/filter/system-ID: identity, mean-bias calibration, affine/ridge affine, learned FIR, ARX, frequency transfer, causal moving average, causal Butterworth, offline Savitzky-Golay, Haar wavelet, and RTS-smoother-style affine smoothing.
- L2 public neural signal baselines: Brossard-style dilated CNN gyro denoising and Calib-Net-style CNN residual calibration.
- L3 public navigation-adapted baselines: RoNIN-style residual ResNet signal adapter, plus metadata-only RoNIN/TLIO/AirIMU references because their original outputs are navigation states rather than high-grade IMU signals.

Outputs include `baseline_metrics.json/csv/md`, per-baseline prediction Parquet files, and `baselines/figures/baseline_summary_grid`, `baseline_gyro_nav_tradeoff`, and `baseline_rank_heatmap` in SVG/PDF/PNG/TIFF. For a faster run, reduce neural baseline training:

```bash
uv run python scripts/evaluate_baselines.py \
  --input data/processed/sim/test.parquet \
  --train data/processed/sim/train.parquet \
  --out outputs/baseline_metrics.json \
  --neural-epochs 1
```

## Training Against ARX/FIR

The default TCN config uses `target_mode: competitor_residual` so the model no longer learns only `low -> high`. It first fits a causal system-ID competitor from the training split, stores that ARX/FIR state in the checkpoint, and trains the TCN to predict a correction on top of that competitor. The validation score also logs relative metrics such as `val_relative_gyro_rmse` and `val_competitor_coherence_gap`.

Key knobs in `configs/train_tcn.yaml`:

- `training.competitor_baseline.mode`: `arx`, `fir`, or `blend`.
- `training.baseline_margin_weight`: penalizes predictions that fail to beat the competitor by the configured margin.
- `training.competitor_correction_scale_gyro` and `training.competitor_correction_scale_accel`: control how much learned correction is applied at validation/inference time.
- `model.zero_init_head: true`: starts the correction head from the competitor baseline.

## SmartPNT-MSF Data

Download or clone the public dataset into `data/raw/smartpnt`. Large files are intentionally ignored by Git.

```bash
uv run python scripts/download_smartpnt.py \
  --repo-url https://gitee.com/lv-jiarui/SmartPNT-MSF-Datasets \
  --out data/raw/smartpnt \
  --manifest configs/smartpnt_manifest.yaml

uv run python scripts/inspect_dataset.py --root data/raw/smartpnt --out outputs/smartpnt_inspection.md
```

If automated clone/download fails, manually place the dataset under `data/raw/smartpnt` and rerun the inspection command. Preparation is config-driven because the reader first inspects files instead of assuming a fixed format:

```bash
uv run python scripts/prepare_smartpnt.py --config configs/smartpnt_example.yaml
```

## Outputs

- `data/raw/`: manually downloaded or cloned original datasets; not committed.
- `data/processed/`: reproducible derived Parquet/HDF5 files; not committed.
- `runs/<experiment_id>/`: source-of-truth configs, checkpoints, predictions, metrics, baseline comparisons, figures, and reports; not committed.
- `outputs/latest/`: convenience copy of the latest completed run; not committed.
- `configs/`: experiment YAML files committed for reproducibility.

## Visualization

All publication figures should be generated through `src/virtual_imu/viz` or the repo skill at `.agents/skills/imu-research-visualization`. Figures export to SVG, PDF, and PNG at 300 dpi with consistent fonts, line widths, captions, and figure indexing.

## FAQ

**Can I run without SmartPNT-MSF?** Yes. `run_smoke_test.py` and the simulation commands are fully self-contained.

**What happens if the SmartPNT format is not recognized?** The inspection and preparation scripts report the unknown file type and continue where possible. They do not assume a fixed schema.

**Are large data or checkpoints committed?** No. `data/raw`, `data/processed`, `runs`, `outputs`, and `checkpoints` are ignored.

**How do I reproduce a run?** Keep the YAML config, generated `data_manifest.json`, run config snapshot, and Git commit hash. The report includes these when available.
