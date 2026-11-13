# smartpnt-virtual-imu

`smartpnt-virtual-imu` is a reproducible research framework for learning a virtual high-grade IMU from low-cost or lower-grade IMU signals. It covers public-dataset preparation, recording-aware splits, paired signal mapping, SmartPNT short-window navigation, neural and classical baselines, cross-domain evaluation, parameter/latency audits, Nature-style figures, and traceable report generation.

The repository tracks source code and frozen protocols. Raw data, prepared tensors, checkpoints, run directories, predictions, figures, and complete reports are external artifacts by design. See [Artifact Policy](docs/ARTIFACTS.md).

**Latest results (2026-11-13):** [ALiMU results and paper conclusions](docs/results/2026-11-13/README.md) now include the shared-model context study and FIR/ARX comparisons on identical targets and information bounds. [Reports, full tables and paper figures](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113) are published as GitHub Release assets with SHA256 provenance.

| ALiMU shared-model condition | FIR S | Recursive ARX S | Neural result S |
|---|---:|---:|---|
| Zero model lookahead, fixed frontend | 0.785716 | 0.871371 | GRU: **0.522061** |
| Six samples / 150 ms model lookahead | 0.711962 | 0.713568 | Transformer backend: **0.372671** |
| Full 320-point window | 0.710244 | 0.710594 | CNN-BiGRU, 40-epoch search: **0.361670** |

S is the equal-device mean of gyro/Raw and accel/Raw RMSE ratios; lower is better and Raw=1. These rows retain their original neural protocols. The 80-epoch window study reaches S=0.355320 with CNN-BiGRU; a separately VALID-selected zero-lookahead frontend configuration reaches S=0.477732. Held-device results depend on the target sensor, and Raw has the lowest three-fold mean. See the result page for selection rules, physical units and source identities.

Public resources:

- SmartPNT-MSF paper: <https://arxiv.org/abs/2507.19079>
- SmartPNT-MSF dataset repo: <https://gitee.com/lv-jiarui/SmartPNT-MSF-Datasets>

## Start Here

- [Latest ALiMU research results](docs/results/2026-11-13/README.md): numerical conclusions, matched classical controls, figures and download checksums.
- [Context/mechanism study](docs/ALIMU_CONTEXT_STUDY.md) and [matched FIR/ARX supplement](docs/ALIMU_CLASSICAL_BASELINES.md): frozen protocols and commands.
- [DDPM matched comparison](docs/MATCHED_DDPM_COMPARISON.md), [shared ablation](docs/CNN_BILSTM_ABLATION.md), and [module replacements](docs/CNN_LSTM_MODULE_REPLACEMENT.md): earlier supporting experiments.
- [Research project framework](docs/PROJECT_FRAMEWORK.md): questions, tasks, architecture, invariants, and directory map.
- [Reproducibility guide](docs/REPRODUCIBILITY.md): environment setup and the staged reproduction workflow.
- [Artifact policy](docs/ARTIFACTS.md): what is versioned, what is external, and how results are identified.
- [Results snapshot](docs/RESULTS_SNAPSHOT.md): benchmark scope and interpretation boundaries.
- [Continuous integration](docs/CI.md): local checks and the token-gated GitHub Actions template.
- [Documentation index](docs/README.md): all protocol, data, matrix, server, and CAD guides.

## Repository Map

| Path | Contents |
|---|---|
| `src/virtual_imu/` | Package modules for IO, preprocessing, models, training, evaluation, inference, and visualization |
| `scripts/` | Reproducible command-line entry points for data, training, analysis, server queues, and release packaging |
| `configs/` | Simulation, public-data, split, model, benchmark, figure, report, and server YAMLs |
| `docs/` | Scientific protocol, data layout, reproducibility, artifact policy, and run guides |
| `tests/` | Unit, contract, provenance, and optional local-data integration tests |
| `.agents/skills/imu-research-visualization/` | Shared style and QA procedure for publication figures |
| `data/` | Local mount point for external datasets; contents are ignored by Git |
| `runs/` | Source-of-truth experiment outputs; contents are ignored by Git |

## Reproduction Ladder

1. **Repository check:** `uv run python scripts/check_repository.py` and `uv run pytest -m "not integration"`.
2. **Self-contained smoke:** `uv run python scripts/run_smoke_test.py`.
3. **Public data:** acquire SmartPNT-MSF, ALiMU, and PennCOSYVIO assets and retain source manifests.
4. **Unified dataset:** align 40 Hz signals, deduplicate archived trajectories, and freeze physical-recording splits.
5. **Bounded experiment:** run the smoke/single-cell profile for the chosen matrix.
6. **Full benchmark:** execute the frozen selection and formal runs without test-set tuning.
7. **Publication:** aggregate saved predictions, generate indexed figures, build the report, and verify external package checksums.

The complete command sequence and evidence contract are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Quick Start

```bash
uv sync --frozen --extra dev
uv run python scripts/check_repository.py
uv run python scripts/run_smoke_test.py
```

The frozen environment targets Python 3.11 or 3.12 with PyTorch 2.4.0 and the CUDA 12.1 wheel index. The smoke test generates a tiny paired IMU simulation, trains a small TCN for 1-2 epochs, runs inference, evaluates metrics, exports paper-style figures, creates a report, and runs `pytest`.

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

`scripts/evaluate_baselines.py` now produces fair train/test baseline comparisons, split into causal-online and noncausal-offline leaderboards:

- L1 classical/filter/system-ID: identity, mean-bias calibration, affine/ridge affine, learned FIR, ARX, frequency transfer, causal moving average, causal Butterworth, offline Savitzky-Golay, Haar wavelet, and RTS-smoother-style affine smoothing.
- L2 public neural signal baselines: Brossard-style dilated CNN gyro denoising and Calib-Net-style CNN residual calibration.
- L3 inertial-odometry references: RoNIN, TLIO, AirIMU, and AirIO are evaluated only through trajectory/state predictions exported by their official or external implementations. They are not treated as high-grade IMU signal generators.

`--train` is required. The evaluator no longer fits trained baselines on the test split because that leaks test information into bias, affine, FIR, ARX, frequency-transfer, and neural competitors.

Outputs include `baseline_metrics.json/csv/md`, `sota_navigation_metrics.csv/md`, per-baseline prediction Parquet files, and `baselines/figures/baseline_summary_grid`, `baseline_gyro_nav_tradeoff`, and `baseline_rank_heatmap` in SVG/PDF/PNG/TIFF. For a faster run, reduce neural baseline training:

```bash
uv run python scripts/evaluate_baselines.py \
  --input data/processed/sim/test.parquet \
  --train data/processed/sim/train.parquet \
  --out outputs/baseline_metrics.json \
  --neural-epochs 1
```

To compare real SOTA inertial odometry outputs, export each method's trajectory/state predictions to CSV/Parquet with `timestamp_s` and one position column triplet such as `position_x_m`, `position_y_m`, `position_z_m`, then pass them as repeatable `--navigation-prediction` inputs:

```bash
uv run python scripts/evaluate_baselines.py \
  --input data/processed/sim/test.parquet \
  --train data/processed/sim/train.parquet \
  --out outputs/baseline_metrics.json \
  --neural-epochs 0 \
  --navigation-prediction outputs/sota/ronin_official.parquet \
  --navigation-prediction outputs/sota/tlio_official.parquet
```

## Training Against ARX/FIR

The default TCN config uses `target_mode: competitor_residual` so the model no longer learns only `low -> high`. It first fits a causal system-ID competitor from the training split, stores that ARX/FIR state in the checkpoint, and trains the TCN to predict a correction on top of that competitor. The validation score also logs relative metrics such as `val_relative_gyro_rmse` and `val_competitor_coherence_gap`.

Key knobs in `configs/train_tcn.yaml`:

- `training.competitor_baseline.mode`: `arx`, `fir`, or `blend`.
- `training.baseline_margin_weight`: penalizes predictions that fail to beat the competitor by the configured margin.
- `training.competitor_correction_scale_gyro` and `training.competitor_correction_scale_accel`: control how much learned correction is applied at validation/inference time.
- `model.zero_init_head: true`: starts the correction head from the competitor baseline.

## SmartPNT-MSF Data

Download or manually place the public dataset under `data/raw/smartpnt_msf`. Large files are intentionally ignored by Git. Keep original archives in `00_archives/` and extract usable folders into `01_extracted/`.

```bash
uv run python scripts/download_smartpnt.py \
  --repo-url https://gitee.com/lv-jiarui/SmartPNT-MSF-Datasets \
  --out data/raw/smartpnt_msf/01_extracted \
  --manifest configs/smartpnt_manifest.yaml

uv run python scripts/inspect_dataset.py --root data/raw/smartpnt_msf/01_extracted --out outputs/smartpnt_msf/inspection.md
```

If automated clone/download fails, manually place the downloaded archives under `data/raw/smartpnt_msf/00_archives`, extract them under `data/raw/smartpnt_msf/01_extracted`, and rerun the inspection command. Preparation is config-driven because the reader first inspects files instead of assuming a fixed format:

```bash
uv run python scripts/prepare_smartpnt.py --config configs/smartpnt_example.yaml
```

For the parsed SmartPNT-MSF IMU/GT alignment, use the GT-supervised dataloader over `data/processed/smartpnt_msf/imu_gt_aligned`:

```bash
python scripts/check_smartpnt_gt_dataloader.py \
  --data-root data/processed/smartpnt_msf/imu_gt_aligned \
  --label-mode pose_delta \
  --window-size 200 \
  --stride 50
```

Inputs are `gx_radps/gy_radps/gz_radps/ax_mps2/ay_mps2/az_mps2`. Label modes are `velocity`, `position`, `velocity_position`, `position_delta`, and `pose_delta`. Do not split duplicated `Traj1`/`Traj2` as independent train/test evidence.

Run a short GT-supervised inertial-odometry training pass with:

```bash
python scripts/train_gt_supervised.py --config configs/smartpnt_gt_supervised.yaml
```

The default config trains `pose_delta` on `Traj3`, validates on `Traj4`, tests on `Traj1`, and skips duplicated `Traj2`.

## Multi-Dataset IMU Pilot

The reproducible pilot catalog downloads one NavINST vehicle sequence, PennCOSYVIO Tango/VI-Sensor data, and selected FusionPortable static/dynamic sequences. Run a source-resolution check first:

```powershell
uv run python scripts/download_pilot_datasets.py `
  --config configs\multi_dataset_pilot_download.yaml `
  --dry-run
```

Remove `--dry-run` to download all three pilots, or repeat `--dataset` to limit the run, for example `--dataset navinst_pilot --dataset penncosyvio_pilot`. HTTP downloads resume from `.part` files. FusionPortable folder resolution runs `gdown 6.1.0` through `uvx`, with its tool cache redirected to `data/raw/_tool_cache/uv`; large Drive files themselves use resumable direct HTTP into E-drive dataset roots.

The authoritative raw-data records are `data/raw/*_pilot/_manifests/pilot_manifest.json`; the batch summary is `data/raw/_pilot_manifests/pilot_summary.json`. Original archives remain unchanged under `00_archives`, while safe extraction is isolated under `01_extracted/<artifact-id>`.

## Strict Multi-Dataset IMU Preparation

The strict pipeline keeps trainable calibrated pairs separate from diagnostic-only data. Run a bounded inspection first; its outputs use a separate `bounded_120s` variant and cannot overwrite a full run:

```powershell
uv run python scripts/prepare_multi_imu_strict.py `
  --config configs\multi_dataset_strict_prepare.yaml `
  --fetch-calibration `
  --dry-run `
  --max-duration-s 120
```

Remove `--dry-run` to acquire the declared Penn/Fusion calibration assets and prepare data. Repeat `--dataset penncosyvio` or `--dataset fusionportable` to restrict a run. NavINST remains diagnostic-only until the official `setup_transformation.txt` is present at `data/raw/navinst_pilot/02_calibration/setup_transformation.txt`; the config records the official repository directory but does not invent a direct file URL.

The source-of-truth artifacts are:

- `data/processed/multi_imu_strict/manifest.json`: eligibility and strict Parquet provenance.
- `data/processed/multi_imu_strict/report.md`: readable per-pair quality table.
- `data/processed/multi_imu_diagnostics/manifest.json`: all passing and blocked diagnostic records.
- `data/processed/multi_imu_strict/splits/`: Penn LOSO and Fusion guarded temporal split manifests.

Diagnostic Parquet is deterministically sampled, while its adjacent JSON retains full-resolution aggregate metrics. Training loaders reject rows explicitly marked `training_eligible: false`. Fusion's `within_sequence_temporal` split measures guarded interpolation within one trajectory; it is not evidence of cross-trajectory or cross-platform generalization.

## Outputs

- `data/raw/`: manually downloaded or cloned original datasets; not committed.
- `data/processed/`: reproducible derived Parquet/HDF5 files; not committed.
- `runs/<experiment_id>/`: source-of-truth configs, checkpoints, predictions, metrics, baseline comparisons, figures, and reports; not committed.
- `outputs/latest/`: convenience copy of the latest completed run; not committed.
- `configs/`: experiment YAML files committed for reproducibility.

## Visualization

All publication figures should be generated through `src/virtual_imu/viz` or the repo skill at `.agents/skills/imu-research-visualization`. Figures export to SVG, PDF, and PNG at 300 dpi with consistent fonts, line widths, captions, and figure indexing.

## Development Checks

```bash
uv run python scripts/check_repository.py
uv run pytest -m "not integration"
uv run ruff check src scripts tests
```

Integration tests that need prepared public data are marked `integration`. GPU-only checks skip automatically when CUDA is unavailable.

## FAQ

**Can I run without SmartPNT-MSF?** Yes. `run_smoke_test.py` and the simulation commands are fully self-contained.

**What happens if the SmartPNT format is not recognized?** The inspection and preparation scripts report the unknown file type and continue where possible. They do not assume a fixed schema.

**Are large data or checkpoints committed?** No. `data/raw`, `data/processed`, `runs`, `outputs`, `checkpoints`, and other generated artifacts are ignored and checked before release.

**How do I reproduce a run?** Keep the YAML config, generated `data_manifest.json`, run config snapshot, and Git commit hash. The report includes these when available.

**Is this a reproduction of every named SOTA system?** No. Published backbones are adapted to a shared task and information scope. The adaptations and omitted original components are documented in the protocol and report.

## Citation and Data Use

Repository metadata is available in [CITATION.cff](CITATION.cff). If you reuse the software or protocol, also cite the upstream SmartPNT-MSF, ALiMU, PennCOSYVIO, and any other dataset you prepare, following each provider's license and attribution requirements.
