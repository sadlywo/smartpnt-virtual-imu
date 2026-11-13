# Matched DDPM Comparison Implementation Plan

> Execute the user-approved experiment continuously using subagent-driven
> development for the independent model adapter task and an independent final
> review. Do not request approval again for the already-approved design.

**Goal:** Complete a controlled ALiMU three-device, three-seed comparison of
DDPM, CNN-BiLSTM, Informer, Transformer and a strict no-diffusion ablation.

**Architecture:** Reuse verified paired data/windows, DDPM equations and common
physical prediction/metrics. Add independent model adapters and a matrix runner
that locks all choices before held-out evaluation, plus aggregate visualization.

**Tech stack:** pinn_imu Python/PyTorch CUDA, YAML, pandas, NumPy, Matplotlib.

- [x] Add model adapters in `src/virtual_imu/models/matched.py` and meaningful
  architecture/ablation tests in `tests/test_matched_models.py`; reuse the
  historical CNN-BiLSTM/Informer implementations with attribution and hashes.
- [x] Add common runner in `src/virtual_imu/diffusion/matched.py`, YAML in
  `configs/ddpm/alimu_matched.yaml`, CLI `scripts/run_matched_comparison.py`.
  Test selection gates, validation support, deterministic sampling and resume.
- [x] Add report/plot module in `src/virtual_imu/viz/plots_matched.py` using
  current paper styling, per-seed tables and exact paired ablation deltas.
- [x] Run targeted tests and CUDA smoke, request independent scientific/code
  review, fix material issues before starting the frozen formal matrix.
- [x] Run all preregistered trials, selected three-seed fits, strict ablation,
  full held-out predictions and CUDA latency. Continue monitoring this turn.
- [x] Independently replay metrics and selection minima, check artifacts and
  figure rendering, document actual findings and limitations in the final report.

Verification commands use `D:/Anaconda3/envs/pinn_imu/python.exe`, PYTHONPATH=src,
PYTHONUTF8=1, and an explicit pytest basetemp under outputs. No commits of runs,
datasets, checkpoint tensors or figures; no remote publication or messaging.

Completed local run: `runs/alimu_matched_ddpm_20261106_final/`. All 41 checks
passed. Independent replay verified 81 fits, 54 final slots, 108 prediction
files, 640 metric rows and 864 recording metric rows. Every one of the 18
strict paired deltas favored removing diffusion. The interpretation is in
`findings.md`; both original and repeated CUDA timings are preserved with
separate hash receipts. Original sealed artifacts remain unchanged.
