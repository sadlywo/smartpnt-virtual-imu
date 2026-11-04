# ALiMU Leave-Sensor Matrix Design

## Goal

Run a reproducible, strict ALiMU experiment matrix that evaluates low-to-high IMU conversion on the unseen `MPU9250` low-cost IMU, while keeping the Penn study's method set, training budget, and result traceability.

## Scope and Claim

The source split is `data/processed/alimu_strict/leave_sensor_mpu9250`.

- Training and validation data may contain only `BMX160` and `LSM9DS1` real sequences.
- Test data may contain only `MPU9250` real sequences.
- The reference target is `VN100` at ALiMU's native 40 Hz rate.
- This is a leave-low-sensor-out evaluation. It supports a claim about generalization to an unseen low-cost IMU model within ALiMU; it is not a cross-dataset or cross-platform transfer claim.

## Experiment Contract

The matrix contains one strict holdout and 15 neural trials:

| Axis | Values |
|---|---|
| Holdout | `MPU9250` |
| Methods | `tcn`, `bias_only`, `drift_only`, `bias_drift`, `fir_bias_drift_residual` |
| Seeds | `20260712`, `20260713`, `20260714` |
| Neural trial count | `5 x 3 = 15` |
| Analytic baselines | `identity`, `mean_bias`, `channel_affine`, `learned_fir_causal` |
| Epoch budget | 30 maximum epochs per neural trial |
| FIR state | 9 taps, ridge alpha 0.01, fitted on train only |

All methods use the same train/validation/test files and standard six-axis targets. ALiMU keeps its native temperature feature and 40 Hz sampling rate; those are dataset properties, not extra budget. The shared window, optimizer, loss weights, early-stopping policy, static-context procedure, and branch architecture must be fixed across the five neural methods except for the declared ablation switches.

## Method Definitions

- `tcn`: ordinary residual TCN, without bias or static-drift branch.
- `bias_only`: constrained low-rank, slow six-axis offset branch; no static-drift branch.
- `drift_only`: static-context drift branch; no bias branch.
- `bias_drift`: both constrained offset and static-drift branches.
- `fir_bias_drift_residual`: frozen train-fitted causal FIR prediction plus zero-initialized `bias_drift` residual correction.

Static context, FIR coefficients, normalizers, and every baseline state must be fit from the training partition only. Validation is used only for checkpoint selection. The held-out MPU9250 test partition must never be used to fit a transform, select a static window, or choose a checkpoint.

### Strict Static-Context Protocol

The ALiMU `drift_only`, `bias_drift`, and `fir_bias_drift_residual` methods use `reference_scope: train_global`, not `train_sensor`. A single deterministic 2048-sample low-motion window is selected from the union of the training split's BMX160 and LSM9DS1 sequences, standardized with train-only feature statistics, and stored under the reserved key `__train_global__` in the checkpoint.

That exact train-derived window is supplied to every training, validation, and held-out MPU9250 inference sequence. No MPU9250 sample, including an unlabeled stationary segment, is inspected to build or replace it. This makes the static branch a transferable training-domain prior rather than an online target-device calibration mechanism. Static-window metadata must record the selected source sensor, source sequence, timestamps, row offset, selection score, and scope; the ALiMU runner must verify that the source sensor is one of `BMX160`/`LSM9DS1` and that the source sequence belongs to the training split.

## Artifacts and Resume Behavior

The runner writes one source-of-truth directory named `runs/alimu_leave_sensor_matrix_<timestamp>/` containing:

- matrix and effective YAML config snapshots;
- input fingerprints and split provenance;
- cached train-fitted baseline states and held-out predictions;
- one trial directory per neural trial, with checkpoint, held-out prediction, metrics, and trial metadata;
- `trial_ledger.csv`, `trials.csv/json/md`, `overall_summary.csv/json/md`, and a Chinese report;
- an active-run pointer written before work begins, and latest-run pointer written only after a complete aggregate.

Resume must reject a changed matrix config, base config, ALiMU split manifest, or completed artifact hash. Completed valid baselines and trials are skipped rather than recomputed.

## Aggregation and Report

Because ALiMU has one strict holdout, seed mean and seed standard deviation are the primary uncertainty statistics; no fabricated fold standard deviation is reported. The aggregate ranks methods by mean normalized RMSE and reports gyro/accelerometer RMSE and MAE, correlation, frequency metrics, phase, coherence, and preintegration metrics when available.

The Chinese report must state the held-out MPU9250 protocol, train-only fitting rule, 15-trial coverage, seed stability, ablation ranking, FIR-residual difference, and limitations. A cross-study paragraph may compare ALiMU and Penn only through within-study relative reduction versus identity/FIR and method-rank stability. It must not compare their absolute RMSE values as though their units, sample rates, references, and datasets were identical.

## Validation

Tests must cover deterministic expansion to exactly 15 unique trial IDs, rejection of invalid sensor partitions or changed inputs, controlled ablation config generation, the train-global static-window fallback and its absence of held-out data, train-only baseline fitting, resume and corruption safeguards, seed aggregation, and report wording/provenance checks. A smoke mode may reduce epochs and sample counts but must preserve the exact method and seed axes. Formal training is run manually by the user and is not started during implementation.
