# Matched Context FIR/ARX Implementation Plan

**Goal:** Complete the user-approved FIR/ARX comparison on the sealed November 8
ALiMU context-study targets, splits and information bounds, in a new run.

**Architecture:** Reconstruct `StudySource` / `CommonWindows` and verify their
support identities against the sealed source. Fit deterministic shared linear
residual models on TRAIN center targets. Select on VALID, lock all states, then
evaluate TEST. Keep the existing source run read-only.

**Tech stack:** NumPy/SciPy, pandas, PyTorch float64 sufficient statistics,
PyYAML, the repository's IMU plotting style, pytest.

## Frozen comparison design

- Shared full-window conditions: W=40,80,160,320 on identical center40 targets.
- Shared bounded-future conditions: W=320, future=0 and6 samples.
- Held-device conditions: W=320/future6 and full, source TRAIN/VALID excludes the target
  device; target TEST uses the same two physical recordings as the source study.
- FIR: multivariate LOW lag regression; ARX-style: the existing LOW lag/delta
  definition; recursive residual ARX: LOW lags and past residual outputs,
  teacher-forced TRAIN fitting and exclusively predicted output feedback at
  validation/test. Reset all state at every input anchor.
- TRAIN-only normalization is copied from the original study. No HIGH input at
  prediction time. Zero padding acts on normalized inputs/residual state.
- Three primary history/regularization packages per family, plus a declared
  3-history by4-alpha sensitivity search. Select primary and expanded results
  separately using VALID S; deterministic fits have n=1, no artificial seed SD.
- Only center40 TRAIN targets enter the normal equations. Full input anchors
  provide context. Never build lags across anchor or recording boundaries.
- Recursive ARX candidates with unstable TRAIN-fitted companion matrices or
  nonfinite free-running validation predictions are recorded as invalid.
- Record retrospective status: historical results on this TEST already exist.

## Tasks

- [x] Add behavioral tests for anchor boundaries, bounded future, recursive
  prediction without observed outputs, sufficient-statistic ridge fitting,
  support mismatch and VALID-only selection. Run them before implementation.
- [x] Implement `context_study/classical_models.py`: bounded lag construction,
  streaming sufficient statistics, centered ridge solutions, free-run prediction.
- [x] Implement `classical_pipeline.py` and CLI: freeze config/source identity,
  fit and lock selection, evaluate matched supports, retain states/predictions,
  independently recompute metrics and check information bounds.
- [x] Implement reporting and figures through `src/virtual_imu/viz`; export
  SVG/PDF/PNG300dpi and captions with explicit comparison conditions.
- [x] Run focused tests and the actual YAML experiment, inspect the figures,
  verify saved predictions/states/metrics/source hashes, and seal the new run.
- [x] Document results, comparison definitions and commands; preserve all
  existing user changes. No dataset, run artifact or figure is committed.

## Verification commands

```powershell
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 -m pytest -q tests/test_context_classical_models.py tests/test_context_classical_pipeline.py
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/run_context_baselines.py --config configs/analysis/alimu_context_classical.yaml
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/run_context_baselines.py --config configs/analysis/alimu_context_classical.yaml --verify
```
