# Conditional DDPM + Transformer implementation plan

**Goal:** Train and tune three independent ALiMU low-cost IMU to VN100 models on local CUDA.

**Architecture:** A conditional Transformer denoises normalized high-minus-low six-axis residuals.
The complete DDPM implements forward Gaussian diffusion, posterior coefficients, x0/epsilon
objectives, and all ancestral reverse steps. Each sensor pair uses the existing frozen,
recording-disjoint unified_imu_v2 split, train-only normalization and affine baseline.

**Tech stack:** PyTorch 2.3.1 CUDA 12.1 in pinn_imu, NumPy, Pandas/Parquet, YAML, Matplotlib.
The installed environment is Python 3.9, so the new entry point avoids existing Python 3.11-only
utility imports and supports direct execution with PYTHONPATH=src; it does not reinstall the package.

- [x] Inspect manifest, sample fields, hardware and existing plotting style; create branch.
- [x] Resolve dataset: user selected ALiMU BMX160, MPU9250 and LSM9DS1 to VN100, trained separately.
- [x] Add numerical tests for forward/reverse diffusion, conditioning, gradients and seeded sampling.
- [x] Implement verified loading, recording leakage checks, train-only scaling and overlap stitching.
- [x] Implement CUDA/AMP Adam training, validation sampling, early stopping and checkpoint restore.
- [x] Run YAML-defined search separately per pair; lock selection before accessing test signals.
- [x] Evaluate raw, affine and selected DDPM on identical held-out rows; export per-axis and grouped
  RMSE/MAE/bias/SNR and improvement, per-record metrics and predictions.
- [x] Generate training, search, improvement, time-series, error and spectral figures using viz styles.
- [x] Save config/manifest/source hashes, environment, checkpoint, metrics, reports and artifact index
  in runs/<experiment_id>; keep all generated outputs untracked and preserve outputs/latest.
- [x] Run relevant tests, inspect real CUDA results and figures, then document measured outcomes.

Final verified run: `runs/alimu_ddpm_transformer_20261106_final/`.
13 targeted tests passed; Ruff passed. The independent artifact audit verified 9 trials,
3 selected models, all validation/test metrics, source/data hashes and 75 figure files.
The earlier `_1530` run and interrupted `_reviewed` run are retained as preliminary records.

Model selection uses equally weighted gyro/accel relative validation RMSE from a fixed validation
window subset and fixed sampling seed, using the same four-draw averaging and overlap stitching
as final testing, with a complete T-step DDPM chain. Full validation is run
for the winning checkpoint and full test data is evaluated only after selection. x0 prediction
with MSE is the default regression parameterization; epsilon prediction is also implemented and
tested. No supervised Transformer results are attributed to the DDPM. Bidirectional attention
and overlapping windows make this an offline mapping experiment, not a causal deployment claim.
