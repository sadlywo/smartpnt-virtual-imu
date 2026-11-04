# Experiments

Recommended experiment order:

1. Run `scripts/run_smoke_test.py` to verify the full loop.
2. Generate the default simulation splits with `configs/sim_baseline.yaml`.
3. Train the TCN baseline with `configs/train_tcn.yaml`.
4. Run the baseline suite with an explicit training split: L1 classical/filter/system-ID, L2 public neural signal adapters, and optional L3 SOTA inertial-odometry prediction imports.
5. Compare the trained virtual IMU model against the causal-online leaderboard first, then use the noncausal-offline leaderboard as an upper-bound reference.
6. After SmartPNT inspection, parse real IMU/GT alignment with `scripts/prepare_smartpnt_msf_real.py`.
7. Check GT-supervised loader shapes with `scripts/check_smartpnt_gt_dataloader.py` and `configs/smartpnt_gt_supervised.yaml`.
8. Train the first GT-supervised inertial-odometry model with `scripts/train_gt_supervised.py --config configs/smartpnt_gt_supervised.yaml`, then compare against zero/mean pose-delta baselines before drawing conclusions.

Every run should preserve:

- Config snapshot
- Git commit hash
- Data manifest
- Training metrics
- Evaluation JSON/CSV/Markdown
- Baseline JSON/CSV/Markdown and baseline figure index
- SOTA navigation CSV/Markdown if external RoNIN/TLIO/AirIMU/AirIO predictions are supplied
- Figure index
- Report

Fairness rules:

- Always pass `--train` to `scripts/evaluate_baselines.py`; trained baselines must never fit on the test split.
- Report causal-online and noncausal-offline baselines separately.
- Original inertial-odometry methods are compared through exported trajectory/state predictions, not through adapted high-grade IMU signal outputs.
