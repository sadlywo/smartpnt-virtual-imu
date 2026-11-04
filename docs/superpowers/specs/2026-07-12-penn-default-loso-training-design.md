# Penn Default LOSO Complete Training Design

## Goal

Train and evaluate a reproducible low-to-high IMU baseline on the strict PennCOSYVIO data using the default leave-one-sequence-out fold:

- train: `af`, `as`
- validation: `bf`
- test: `bs`

The experiment compares the existing two-branch bias/drift TCN and a plain TCN against Identity, train-only mean bias, train-only affine calibration, and train-only causal FIR. The test sequence is never used for fitting, normalization, model selection, static-context selection, or early stopping.

## Source Of Truth

The authoritative split is:

`data/processed/multi_imu_strict/splits/penn_loso/default_bs.json`

Training reads the immutable strict Parquet paths referenced by this manifest. It does not materialize duplicate train, validation, or test Parquet files. Every referenced entry must be `training_eligible=true`, and the loader validates that sequence groups do not overlap.

## Input And Target Contract

The model input contains only the calibrated low-side six axes:

`gx_low, gy_low, gz_low, ax_low, ay_low, az_low`

The all-null Penn temperature column is excluded. Targets are the corresponding six high-grade axes. Model training uses residual mode (`high - low`) so the prediction remains anchored to the physical low-grade signal.

Feature and target normalization statistics are fit on `af/as` only and reused unchanged for `bf/bs`.

## Two-Branch Context Contract

### Bias branch

The bias branch keeps the existing constrained offset form: each input window produces one six-axis offset vector, which is broadcast over time. It cannot emit an arbitrary time-varying hidden correction.

Penn does not provide authoritative stationary labels, so the auxiliary stationary bias loss is disabled for this main experiment. This avoids creating pseudo-targets from unverified motion labels. The branch is still trained through the supervised residual objective.

### Drift branch

The drift branch receives one long, contiguous low-motion context selected from training data only. Selection uses low-side IMU values and never reads high-grade targets:

1. Evaluate candidates independently inside each training sequence so timestamps from different sequences are never interleaved.
2. Use a fixed context length of 2,048 samples.
3. Score each candidate by low-side gyroscope energy plus accelerometer-norm instability.
4. Select the minimum-score candidate across `af/as`.
5. Reuse that training-derived sensor context unchanged for validation and test.

This represents a train-time sensor characterization prior, not a transductive search over the held-out trajectory.

## Models

### Plain TCN

- six input channels and six output channels
- residual target
- same hidden width, dilation depth, kernel size, optimizer, loss, and epoch budget as the two-branch model
- no bias or drift branch

### Two-branch TCN

- same main TCN backbone
- constrained six-axis offset bias branch
- long-context drift branch with six static channels
- no stationary auxiliary loss in the Penn main experiment

Both models use seed `20260712`, 30 maximum epochs, validation-score checkpointing, early stopping, and the existing composite signal loss.

## Classical Baselines

All fitted baselines use `af/as` only:

- **Identity:** low signal unchanged; no fitting.
- **Mean bias:** one train-set six-axis mean residual.
- **Affine:** six independent train-set linear mappings with intercepts.
- **Causal FIR:** train-only causal finite impulse response mapping. Filter order and regularization are recorded in the result manifest.

No baseline may fit parameters on `bf` or `bs`.

## Evaluation

Validation (`bf`) selects checkpoints and early stopping. Final ranking uses only test (`bs`). At minimum, the comparison table reports:

- gyroscope and accelerometer RMSE and MAE in physical units;
- mean six-axis correlation;
- percentage RMSE improvement relative to Identity;
- gyroscope spectral error and coherence;
- preintegration error over the configured horizon;
- parameter count and inference class (`causal-window` or `causal-filter`).

Predictions for every model and baseline are saved with timestamps and the original low/high axes so metrics can be recomputed independently.

## Artifact Layout

Each neural run remains under `runs/<experiment_id>/` and contains:

- source config and split-manifest snapshot;
- provenance including git state and strict input fingerprints;
- normalization statistics;
- best checkpoint and epoch metrics;
- `test_predictions.parquet` and `test_metrics.json`.

The comparison output is written under a dedicated Penn LOSO comparison directory with JSON, CSV, and Markdown tables plus a readable report that points back to both neural run directories.

## Failure Rules

Training stops before optimization when:

- the manifest has group overlap or references noneligible rows;
- a referenced Parquet or fingerprint is missing;
- any selected model feature is non-finite;
- train-only static context cannot provide 2,048 contiguous samples;
- validation or test windows are empty;
- a fitted baseline attempts to use non-training rows.

The implementation does not replace missing temperature with a fabricated value for this experiment.

## Verification

Unit tests cover manifest loading, group isolation, six-channel feature selection, train-only context reuse, deterministic low-motion selection, baseline fit boundaries, and test-metric serialization. A short one-epoch smoke run must pass before the two 30-epoch runs begin. Final verification reruns focused tests, Ruff, manifest parsing, prediction schema checks, and comparison-table consistency.
