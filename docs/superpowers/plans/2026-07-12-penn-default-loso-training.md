# Penn Default LOSO Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` to execute this plan task-by-task with review checkpoints.

**Goal:** Train and compare Identity, mean-bias, channel-wise affine, causal FIR, plain TCN, and bias/drift two-branch TCN on the strict Penn default LOSO split (`af/as` train, `bf` validation, `bs` test) without test-target or static-context leakage.

**Architecture:** Add a manifest-native data boundary that validates the strict aggregate manifest before loading any frame. Build the two neural datasets from the same split bundle, select the two-branch model's long low-motion context from train sequences only, serialize it in the checkpoint, and reuse it unchanged for validation and test. Fit all analytic baselines on train rows only and evaluate every method through one common physical-unit metric/report path.

**Tech Stack:** Python 3.10+, PyTorch, pandas/pyarrow, NumPy, SciPy, PyYAML, Typer, pytest, Ruff; execution in the Conda environment `pinn_imu`.

---

## Task 1: Add the manifest-native split loader

**Files:**
- Create: `src/virtual_imu/io/training_split_manifest.py`
- Create: `tests/test_training_split_manifest.py`
- Reference: `src/virtual_imu/io/multi_imu_splits.py`
- Reference: `data/processed/multi_imu_strict/splits/penn_loso/default_bs.json`

1. Write failing tests for loading `train`, `validation`, and `test` records into three frames while preserving `dataset_id`, `pair_id`, and `sequence_id`.
2. Add failure cases for a missing Parquet path, a strict-manifest fingerprint mismatch, an ineligible row, and overlapping group identities across splits.
3. Run:

   ```powershell
   conda run -n pinn_imu python -m pytest tests/test_training_split_manifest.py -q
   ```

   Expected: failures because the loader does not exist.

4. Implement:

   ```python
   @dataclass(frozen=True)
   class TrainingSplitBundle:
       frames: dict[str, pd.DataFrame]
       split_manifest: dict[str, Any]
       strict_manifest: dict[str, Any]
       split_manifest_path: Path
       strict_manifest_path: Path
       input_records: tuple[dict[str, Any], ...]

   def load_training_split_manifest(
       split_manifest_path: Path,
       strict_manifest_path: Path,
   ) -> TrainingSplitBundle:
       ...
   ```

5. Resolve project-relative paths against the repository root, call `validate_no_split_leakage`, require each split record to match the strict manifest by dataset/pair/sequence/fingerprint, and reject any loaded row whose `training_eligible` is not true.
6. Support `row_filter: {type: all}` now and bounded `time_start_s`/`time_end_s` filters so the loader remains compatible with temporal manifests.
7. Run the test command again. Expected: all loader tests pass.

## Task 2: Make static context train-only and low-motion selected

**Files:**
- Modify: `src/virtual_imu/training/dataset.py`
- Create: `tests/test_training_static_context.py`

1. Write failing tests proving that a deterministic selector chooses a planted low-motion interval, never uses high-grade target columns, never joins samples across sequence boundaries, and returns the same start index after high-target values are mutated.
2. Write a failing dataset test showing that validation receives the exact train-derived static window rather than rebuilding one from validation rows.
3. Run:

   ```powershell
   conda run -n pinn_imu python -m pytest tests/test_training_static_context.py -q
   ```

   Expected: failures for the missing selection mode and external static-window injection.

4. Add `selection: auto_low_motion` to `build_static_context_windows`. Score every full-length candidate inside each train sequence using only low-side signals:

   ```python
   gyro_energy = mean(gx_low**2 + gy_low**2 + gz_low**2)
   accel_instability = var(sqrt(ax_low**2 + ay_low**2 + az_low**2))
   score = robust_scale(gyro_energy) + robust_scale(accel_instability)
   ```

5. Select the deterministic minimum by `(score, sequence_id, start_row)` and create one reusable six-axis train context per low sensor. Do not inspect `*_high`, targets, or test/validation data.
6. Extend `IMUWindowDataset` with optional `static_windows`. When `reference_scope: train_sensor`, training may construct the windows; validation/test must receive them and must fail clearly when a sensor context is missing.
7. Preserve existing ALiMU labeled-stationary behavior when the new selection mode is absent.
8. Run the test command again. Expected: all static-context tests pass.

## Task 3: Integrate manifests and context serialization into training

**Files:**
- Modify: `src/virtual_imu/training/trainer.py`
- Create: `tests/test_training_manifest_smoke.py`

1. Write a one-epoch failing smoke test using tiny strict Parquet fixtures and split manifests. Assert that training reads `af/as`, validates on `bf`, excludes `bs`, and accepts six low-side feature columns without temperature.
2. Add assertions that the run directory contains split/strict manifest snapshots and that `best.ckpt` contains the train-derived static windows for the two-branch model.
3. Run:

   ```powershell
   conda run -n pinn_imu python -m pytest tests/test_training_manifest_smoke.py -q
   ```

   Expected: failure because `train_model` only accepts `train_path`/`val_path`.

4. Add `_load_training_frames(data_cfg)` with two compatible modes:
   - New: `split_manifest` plus `strict_manifest`.
   - Legacy: existing `train_path` plus `val_path`.
5. Validate that all six configured low-side feature columns are present and finite. Fit normalization on train only and reuse its statistics unchanged for validation.
6. Construct train static windows once, inject them into validation, and add these keys to checkpoints:

   ```python
   checkpoint["static_windows"] = train_dataset.static_windows
   checkpoint["split_provenance"] = {
       "train": ["af", "as"],
       "validation": ["bf"],
       "test": ["bs"],
   }
   ```

7. Snapshot both manifests into the run directory and include their source paths/fingerprints in provenance.
8. Run the smoke test again. Expected: pass and one checkpoint produced.

## Task 4: Enforce train-context inference and add frame inference

**Files:**
- Modify: `src/virtual_imu/inference/virtual_imu_generator.py`
- Create: `tests/test_train_context_inference.py`

1. Write failing tests that load a checkpoint with `reference_scope: train_sensor`, infer on a `bs` frame, and assert that the serialized train context is used unchanged.
2. Add a failure test for a missing checkpoint context and a schema test for the generated high-grade prediction columns.
3. Run:

   ```powershell
   conda run -n pinn_imu python -m pytest tests/test_train_context_inference.py -q
   ```

4. Load `static_windows` from the checkpoint. For `train_sensor` scope, prohibit rebuilding context from inference data and raise a descriptive error if no matching train context exists.
5. Add:

   ```python
   def run_frame(self, frame: pd.DataFrame, out_path: Path | None = None) -> pd.DataFrame:
       ...
   ```

   Keep `run(input_path, out_path)` as a thin backwards-compatible Parquet wrapper.
6. Run the tests again. Expected: all inference tests pass.

## Task 5: Add leakage-safe Penn baselines and common metrics

**Files:**
- Create: `src/virtual_imu/eval/penn_loso_comparison.py`
- Create: `tests/test_penn_loso_comparison.py`
- Reuse: `src/virtual_imu/eval/baselines.py`
- Reuse: `src/virtual_imu/eval/reports.py`

1. Write failing tests for four train-fitted predictors: Identity, mean bias, six independent affine mappings with intercepts, and causal learned FIR.
2. Assert that mutating test targets cannot change any fitted state, channel-wise affine has no cross-axis coefficients, and FIR history resets at every sequence boundary.
3. Assert that common evaluation returns gyro/accel RMSE and MAE, mean per-axis correlation, improvement versus Identity, spectral/coherence metrics, and preintegration metrics.
4. Run:

   ```powershell
   conda run -n pinn_imu python -m pytest tests/test_penn_loso_comparison.py -q
   ```

5. Implement focused APIs:

   ```python
   def fit_penn_baselines(train_df: pd.DataFrame, fir_taps: int, alpha: float) -> dict[str, Any]:
       ...

   def predict_penn_baselines(
       state: dict[str, Any], test_df: pd.DataFrame
   ) -> dict[str, np.ndarray]:
       ...

   def evaluate_penn_method(prediction_df: pd.DataFrame) -> dict[str, Any]:
       ...

   def write_penn_comparison(...):
       ...
   ```

6. Fit the affine slope/intercept independently for each axis. Reuse the existing multivariate causal FIR state, but expose only `learned_fir_causal` in this experiment.
7. Write one prediction Parquet and one metrics JSON per method plus `comparison.csv`, `comparison.md`, and a machine-readable comparison JSON. Include parameter count and causal/inference class.
8. Run the tests again. Expected: all comparison tests pass.

## Task 6: Add reproducible Penn configs and the experiment runner

**Files:**
- Create: `configs/penn_loso_tcn_default.yaml`
- Create: `configs/penn_loso_bias_drift_tcn_default.yaml`
- Create: `configs/penn_loso_comparison_default.yaml`
- Create: `src/virtual_imu/training/penn_loso_runner.py`
- Create: `scripts/run_penn_loso_comparison.py`
- Create: `tests/test_penn_loso_runner.py`

1. Write failing tests for config validation, CLI `--help`, and a `--smoke` execution that limits each neural model to one epoch but still emits all six method rows.
2. Run:

   ```powershell
   conda run -n pinn_imu python -m pytest tests/test_penn_loso_runner.py -q
   ```

3. Define both neural configs with the same six input axes, residual target, TCN backbone width/depth, optimizer, scheduler, batch/window settings, seed `20260712`, maximum 30 epochs, validation early stopping, and best-validation checkpoint selection.
4. In the two-branch config enable a 2048-sample `auto_low_motion` train-sensor context, six-channel offset bias branch, and drift branch. Disable stationary auxiliary bias loss because Penn has no authoritative stationary labels.
5. Define comparison config entries for the two model configs and FIR hyperparameters. The runner must:
   - Validate/load the strict split once.
   - Fit analytic baselines on `af/as` only.
   - Train TCN and two-branch TCN using `bf` only for validation.
   - Restore each best checkpoint and predict `bs` only.
   - Evaluate all six methods through the common evaluator.
   - Write config snapshots, manifest snapshots, provenance, checkpoints, predictions, metrics, and the final table under one source-of-truth comparison run.
6. Add structured logs for every input read, output written, and skipped artifact.
7. Run the runner tests again. Expected: the smoke run contains rows for `identity`, `mean_bias`, `channel_affine`, `learned_fir_causal`, `tcn`, and `bias_drift_tcn`.

## Task 7: Execute smoke, full LOSO training, and verification

**Files:**
- Generated only: `runs/penn_loso_default_comparison_*/`
- Convenience snapshot only: `outputs/latest/` and `outputs/latest_run.txt`

1. Run focused tests and lint:

   ```powershell
   conda run -n pinn_imu python -m pytest tests/test_training_split_manifest.py tests/test_training_static_context.py tests/test_training_manifest_smoke.py tests/test_train_context_inference.py tests/test_penn_loso_comparison.py tests/test_penn_loso_runner.py -q
   conda run -n pinn_imu python -m ruff check src scripts tests
   ```

   Expected: all tests pass and Ruff reports no errors.

2. Run the one-epoch end-to-end smoke experiment:

   ```powershell
   conda run -n pinn_imu python scripts/run_penn_loso_comparison.py --config configs/penn_loso_comparison_default.yaml --smoke
   ```

   Expected: six methods, test sequence exactly `bs`, and no non-finite metrics/predictions.

3. Inspect smoke provenance before the long run. Require:
   - Train sequences exactly `af`, `as`.
   - Validation sequence exactly `bf`.
   - Test sequence exactly `bs`.
   - Static context source in `af` or `as` only.
   - Baseline fit provenance contains no `bf` or `bs`.

4. Run the full default LOSO experiment:

   ```powershell
   conda run -n pinn_imu python scripts/run_penn_loso_comparison.py --config configs/penn_loso_comparison_default.yaml
   ```

   Expected: both neural models complete at most 30 epochs, restore their best validation checkpoints, and produce `bs` predictions.

5. Run the full test suite and compilation check:

   ```powershell
   conda run -n pinn_imu python -m pytest -q
   conda run -n pinn_imu python -m compileall -q src scripts tests
   conda run -n pinn_imu python -m ruff check src scripts tests
   ```

6. Audit the final comparison artifacts against `outputs/latest_run.txt`; never infer the run identity from `outputs/latest/` alone. Report the exact run directory, epoch selected for each neural model, strict split provenance, and the complete six-method table.

## Completion Criteria

- The strict Penn default LOSO experiment is reproducible from one comparison YAML.
- No validation/test target or test-derived static context is used for fitting.
- Both neural models share the same primary input/backbone/training budget.
- Every method is evaluated on identical `bs` rows in physical units.
- The final run contains traceable manifests, configs, checkpoints, predictions, metrics, and a readable comparison report.
- Focused tests, full tests, compileall, and Ruff all pass.
- No raw/processed data, checkpoints, run directories, or generated figures are committed.
