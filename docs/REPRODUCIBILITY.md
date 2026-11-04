# Reproducibility Guide

## Environment

The frozen benchmark environment targets Python 3.11 or 3.12. The current lock file pins PyTorch 2.4.0 with the CUDA 12.1 wheel index and NumPy 1.26.4. Linux and Windows x86-64 are supported by the declared dependency source; CPU-only execution remains available for smoke tests.

```bash
uv python install 3.12
uv sync --frozen --extra dev
uv run python scripts/check_repository.py
uv run pytest -m "not integration"
```

`pytest` tests marked `integration` require local public-dataset manifests. CUDA-dependent checks skip automatically when no GPU is available.

## Reproduction levels

### Level 0: repository integrity

```bash
uv run python scripts/check_repository.py
uv run pytest -m "not integration"
```

This validates the package layout, YAML syntax, ignored artifact boundary, tracked-file size policy, and obvious credential patterns.

### Level 1: self-contained simulation

```bash
uv run python scripts/run_smoke_test.py
```

The smoke test generates a synthetic paired IMU dataset, runs a short TCN training and inference cycle, evaluates signal and baseline metrics, exports a figure/report, and runs the test suite. It requires no public dataset download.

For an explicit simulation run:

```bash
uv run python scripts/make_sim_data.py --config configs/sim_baseline.yaml
uv run python scripts/run_experiment.py \
  --train-config configs/train_tcn.yaml \
  --test data/processed/sim/test.parquet \
  --baseline-train data/processed/sim/train.parquet \
  --baseline-neural-epochs 4
```

### Level 2: acquire and inspect public data

SmartPNT-MSF:

```bash
uv run python scripts/download_smartpnt.py \
  --repo-url https://gitee.com/lv-jiarui/SmartPNT-MSF-Datasets \
  --out data/raw/smartpnt_msf/01_extracted \
  --manifest configs/smartpnt_manifest.yaml
uv run python scripts/inspect_dataset.py \
  --root data/raw/smartpnt_msf/01_extracted \
  --out outputs/smartpnt_msf/inspection.md
```

Multi-dataset pilot:

```bash
uv run python scripts/download_pilot_datasets.py \
  --config configs/multi_dataset_pilot_download.yaml \
  --dry-run
```

Remove `--dry-run` only after reviewing destinations and volume. Dataset manifests are written next to the raw assets and remain outside Git.

### Level 3: prepare calibrated data and splits

```bash
uv run python scripts/prepare_multi_imu_strict.py \
  --config configs/multi_dataset_strict_prepare.yaml \
  --dry-run \
  --max-duration-s 120

uv run python scripts/prepare_unified_imu.py \
  --config configs/unified_imu_dataset.yaml

uv run python scripts/check_unified_training.py \
  --config configs/check_unified_training.yaml
```

The full preparation should be accepted only when the prepared-data manifest, trajectory deduplication ledger, physical-recording split memberships, and channel/unit schema pass inspection. Do not begin formal test evaluation while these checks are unresolved.

### Level 4: run a bounded experiment

Always use a smoke or single-cell configuration before launching a matrix.

```bash
uv run python scripts/run_formal_training.py \
  --config configs/train_formal_paired.yaml \
  --smoke

uv run python scripts/run_alimu_leave_sensor_matrix.py \
  --config configs/alimu_leave_sensor_matrix_3seed.yaml \
  --smoke

uv run python scripts/run_penn_loso_matrix.py \
  --config configs/penn_loso_matrix_3seed.yaml \
  --smoke

uv run python scripts/run_reviewer_hpo.py \
  --config configs/reviewer_hpo.yaml \
  --plan-only
```

The complete submitted benchmark is substantially larger than the smoke profile. It combines frozen hyperparameter search, three optimization seeds, paired and navigation tasks, shared classical controls, cross-recording transfer, downstream diagnostics, and parameter/latency audits. The exact frozen protocol is in [TRAINING_PROTOCOL_V2.md](TRAINING_PROTOCOL_V2.md) and [BENCHMARK_EXPANSION_20261024.md](BENCHMARK_EXPANSION_20261024.md).

### Level 5: aggregate, evaluate, and report

```bash
uv run python scripts/aggregate_reviewer_results.py \
  --pipeline-root <pipeline-run> \
  --out <summary-run>

uv run python scripts/build_reviewer_figures.py \
  --results-root <immutable-result-package> \
  --out <figure-run> \
  --config configs/reviewer_figures.yaml

uv run python scripts/build_reviewer_report.py \
  --results-root <immutable-result-package> \
  --figures-root <figure-run> \
  --out <report-run> \
  --config configs/reviewer_report.yaml
```

The server-side queue and packaging tools live under `scripts/server/` and `scripts/reviewer_artifact_package.py`. Large benchmark runs should be transferred as checksum-verified packages; never infer their source from `outputs/latest/`.

## Evidence to preserve

Every run must retain:

- the exact YAML config and a config snapshot;
- the Git commit and source fingerprint;
- raw and prepared data manifest hashes;
- split membership and trajectory deduplication records;
- training/validation histories, selected epoch, and checkpoint hashes;
- test predictions and metrics without post-test hyperparameter changes;
- baseline predictions and metrics;
- parameter and latency measurement context;
- figure-source CSVs and figure-index metadata;
- the generated Markdown/HTML/PDF report and page-level QA evidence when publication figures are claimed.

See [ARTIFACTS.md](ARTIFACTS.md) for the directory and release contract.
