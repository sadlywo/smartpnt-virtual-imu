# Penn Four-Fold Ablation and FIR-Residual Design

## Objective

Evaluate whether the constrained bias branch and train-only long-context drift branch generalize across every Penn sequence, and determine whether they add value beyond a strong frozen causal FIR mapping.

## Strict Folds

Use the existing immutable strict Penn manifests. No row-level repartitioning is allowed.

| Fold | Train | Validation | Test | Manifest |
|---|---|---|---|---|
| test_af | as, bf | bs | af | `data/processed/multi_imu_strict/splits/penn_loso/test_af.json` |
| test_as | bf, bs | af | as | `data/processed/multi_imu_strict/splits/penn_loso/test_as.json` |
| test_bf | af, bs | as | bf | `data/processed/multi_imu_strict/splits/penn_loso/test_bf.json` |
| default_bs | af, as | bf | bs | `data/processed/multi_imu_strict/splits/penn_loso/default_bs.json` |

Every fit, normalization statistic, FIR state, and static-context selection must use the fold's train split only. Validation is used only for checkpoint selection and early stopping. Test is evaluated once from the restored best checkpoint.

## Seeds and Budget

- Paired seeds: `20260712`, `20260713`, `20260714`.
- Maximum neural budget: 30 epochs per trial with the existing validation early stopping policy.
- All neural methods within a fold/seed use identical primary TCN width, depth, windows, optimizer, scheduler, losses, and data ordering seed.
- Total neural trials: 4 folds × 5 methods × 3 seeds = 60.
- Analytic baselines are deterministic and fitted once per fold.

## Methods

### Analytic baselines

1. Identity.
2. Train mean bias.
3. Train-fitted independent channel affine.
4. Train-fitted 9-tap causal FIR with ridge alpha 0.01; history resets at sequence boundaries.

### Neural ablation matrix

1. `tcn`: plain residual TCN, target `high - low`.
2. `bias_only`: same TCN plus constrained per-window six-axis offset branch; no drift branch.
3. `drift_only`: same TCN plus train-only 2048-sample low-motion context branch; no bias branch.
4. `bias_drift`: both constrained offset and train-only drift branches.
5. `fir_bias_drift_residual`: frozen train-fitted causal FIR plus the full bias/drift TCN learning `high - FIR(low)`.

The Penn data have no authoritative stationary labels, so stationary auxiliary bias loss remains disabled for all Penn trials. Bias is constrained structurally by the six-axis per-window offset output.

## Frozen FIR-Residual Contract

For each fold and seed, fit the FIR from the fold's train frame only. Serialize that state in the neural checkpoint. The FIR is not optimized by gradient descent.

During training and validation:

```text
fir_prediction = frozen_fir(low)
network_target = high - fir_prediction
final_prediction = fir_prediction + network_correction
```

During test inference, restore both the frozen train FIR state and the best validation-selected network checkpoint. FIR lag history resets for each test sequence. Test targets are never accessed before final metric calculation.

## Static Context Contract

- Context length: 2048 samples.
- Candidate search is performed independently inside each train sequence.
- Score uses low-side gyro energy and acceleration-norm instability only.
- Deterministic minimum is selected across the fold's train sequences.
- The selected normalized window and metadata are serialized in the checkpoint.
- Validation and test must reuse this train-derived window; rebuilding from their own rows is forbidden.
- `bias_only` and plain `tcn` do not consume static context.

## Metrics and Ranking

Each trial emits physical-unit gyro/accel RMSE and MAE, mean per-axis correlation, spectral/coherence/phase-delay metrics, and preintegration residuals.

Primary aggregation is hierarchical:

1. Average three seeds within each fold and method.
2. Macro-average the four fold means so sequence length cannot dominate.
3. Report fold standard deviation, mean within-fold seed standard deviation, and worst fold.

Primary rank uses the equal-weight mean of gyro and accel RMSE ratios to the fold's Identity baseline. Lower is better. Supplementary tables report improvements relative to Identity, plain TCN, and frozen FIR.

Analytic baselines aggregate over four folds only and do not receive artificial seed replication.

## Artifacts

The source-of-truth directory is `runs/<matrix_experiment_id>/` and contains:

- comparison config and all effective trial configs;
- strict and split manifest snapshots with hashes;
- one directory per fold/method/seed with checkpoint, metrics, predictions, and provenance;
- deterministic baseline state/predictions/metrics per fold;
- `trials.csv`, `fold_summary.csv`, `overall_summary.csv`, and JSON equivalents;
- ranked Markdown comparison tables;
- a Chinese report explaining effectiveness, fold instability, FIR gap, navigation-metric trade-offs, and next recommendations.

`outputs/latest/` is only a lightweight convenience snapshot. `outputs/latest_run.txt` remains the authoritative pointer to the source run.

## Failure Handling and Resume

- Validate all four manifests and expected sequence assignments before starting trial 1.
- Generate a deterministic trial ID from fold, method, and seed.
- If a completed trial has matching config/manifest fingerprints, resume by skipping it and logging `SKIPS`.
- A missing/corrupt checkpoint, non-finite prediction, wrong test sequence, or provenance mismatch fails that trial and prevents final aggregation.
- Do not silently substitute another seed, fold, context, checkpoint, or FIR state.

## Acceptance Criteria

- Exactly 60 valid neural trials and 16 deterministic fold-baseline evaluations are represented in the final artifact set.
- Every neural test prediction contains exactly the held-out fold sequence and finite six-axis values.
- Every checkpoint proves train-only normalization, FIR fitting, and static-context provenance where applicable.
- All five neural methods use paired seeds and identical shared backbone/training budgets.
- Hierarchical aggregate tables and the Chinese report are generated from trial artifacts, not manually transcribed values.
- Focused tests, full tests, compileall, and Ruff on touched files pass.
- Raw data, processed data, checkpoints, predictions, and run artifacts remain uncommitted.
