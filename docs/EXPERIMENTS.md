# Experiments

Recommended experiment order:

1. Run `scripts/run_smoke_test.py` to verify the full loop.
2. Generate the default simulation splits with `configs/sim_baseline.yaml`.
3. Train the TCN baseline with `configs/train_tcn.yaml`.
4. Run the three-tier baseline suite: L1 classical/filter/system-ID, L2 public neural signal adapters, and L3 navigation-adapted references.
5. Compare the trained virtual IMU model against the baseline table and the Nature-style baseline figures.
6. After SmartPNT inspection, prepare real aligned windows and repeat training/evaluation.

Every run should preserve:

- Config snapshot
- Git commit hash
- Data manifest
- Training metrics
- Evaluation JSON/CSV/Markdown
- Baseline JSON/CSV/Markdown and baseline figure index
- Figure index
- Report
