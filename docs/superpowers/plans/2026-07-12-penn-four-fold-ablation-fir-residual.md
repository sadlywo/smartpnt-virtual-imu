# Penn Four-Fold Ablation and FIR-Residual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a resumable 4-fold × 5-neural-method × 3-seed Penn experiment matrix with leakage-safe analytic baselines, frozen FIR-residual learning, hierarchical aggregation, and a Chinese report.

**Architecture:** Keep the existing single-fold trainer and inference generator as the trial engine. Add a matrix planner that produces deterministic fold/method/seed trials, a focused matrix runner that validates and resumes those trials, and a separate aggregation module that derives fold and overall tables from immutable trial artifacts. Correct the shared system-ID lag construction so FIR/ARX history resets at every sequence boundary before using it as the hybrid model's frozen primary mapping.

**Tech Stack:** Python, PyTorch, pandas/pyarrow, NumPy, SciPy, PyYAML, argparse, pytest, Ruff; Conda environment `pinn_imu`; CUDA on the RTX 3060 Ti for full trials.

---

## File Responsibilities

- `src/virtual_imu/eval/baselines.py`: sequence-safe system-ID feature construction used by frozen FIR training/inference.
- `src/virtual_imu/training/penn_matrix_config.py`: validate the matrix YAML and create deterministic trial specifications/effective model configs.
- `src/virtual_imu/training/penn_matrix_runner.py`: execute/resume baselines and neural trials; enforce artifact and provenance contracts.
- `src/virtual_imu/eval/penn_matrix_aggregation.py`: read trial metrics and compute seed-within-fold and fold-macro summaries/ranks.
- `src/virtual_imu/eval/penn_matrix_report.py`: render the Chinese report from aggregate JSON/CSV only.
- `scripts/run_penn_loso_matrix.py`: CLI for `--smoke`, `--resume-run`, and full execution.
- `configs/penn_loso_matrix_3seed.yaml`: source-of-truth folds, seeds, methods, shared hyperparameters, baselines, and outputs.
- `tests/test_system_id_sequence_boundaries.py`: FIR/ARX sequence-reset regressions.
- `tests/test_penn_matrix_config.py`: matrix expansion and method override tests.
- `tests/test_penn_matrix_aggregation.py`: hierarchical statistics/ranking tests.
- `tests/test_penn_matrix_runner.py`: smoke, resume, provenance, and corruption tests.

No Git commit is created because the user explicitly requested preserving the current uncommitted workspace.

### Task 1: Make Shared FIR/ARX Features Sequence-Safe

**Files:**
- Modify: `src/virtual_imu/eval/baselines.py`
- Create: `tests/test_system_id_sequence_boundaries.py`

- [ ] **Step 1: Write failing fit/predict boundary tests**

```python
def test_system_id_prediction_resets_lag_history_per_sequence():
    first = paired_frame("af", low_start=100.0)
    second = paired_frame("as", low_start=-100.0)
    combined = pd.concat([first, second], ignore_index=True)
    state = fit_system_id_baseline_state(combined, fir_taps=3, alpha=1e-6)

    together = predict_system_id_from_state(state, combined)["learned_fir_causal"]
    separate = np.vstack([
        predict_system_id_from_state(state, first)["learned_fir_causal"],
        predict_system_id_from_state(state, second)["learned_fir_causal"],
    ])
    np.testing.assert_allclose(together, separate)
```

Also fit once with concatenated train sequences and once with explicitly stacked per-sequence lag features; assert identical FIR weights.

- [ ] **Step 2: Run the tests and confirm the current cross-boundary failure**

Run:

```powershell
$env:PYTHONPATH='src'
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_system_id_sequence_boundaries.py -q
```

Expected: the second sequence's first `fir_taps - 1` predictions differ because history leaks from the previous sequence.

- [ ] **Step 3: Implement one sequence-aware feature helper**

```python
def _sequence_lag_features(df: pd.DataFrame, taps: int, include_delta: bool = False) -> np.ndarray:
    if "sequence_id" not in df.columns:
        return _lag_features(df[LOW_COLUMNS].to_numpy(float), taps, include_delta)
    chunks = []
    for _, group in df.groupby("sequence_id", sort=False):
        ordered = group.sort_values("timestamp_s", kind="stable")
        chunks.append(_lag_features(ordered[LOW_COLUMNS].to_numpy(float), taps, include_delta))
    return np.vstack(chunks)
```

Use it in `fit_system_id_baseline_state`. In `predict_system_id_from_state`, allocate output arrays and predict each contiguous sequence independently so returned rows remain aligned with the input frame.

- [ ] **Step 4: Run focused and existing baseline tests**

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_system_id_sequence_boundaries.py tests/test_supervised_baseline.py tests/test_penn_loso_comparison.py -q
```

Expected: all pass; FIR history resets at sequence boundaries.

### Task 2: Define and Validate the 60-Trial Matrix

**Files:**
- Create: `src/virtual_imu/training/penn_matrix_config.py`
- Create: `tests/test_penn_matrix_config.py`
- Create: `configs/penn_loso_matrix_3seed.yaml`

- [ ] **Step 1: Write failing matrix schema tests**

```python
def test_matrix_expands_to_sixty_paired_trials():
    config = load_config("configs/penn_loso_matrix_3seed.yaml")
    trials = build_trial_specs(config, project_root=Path.cwd())
    assert len(trials) == 60
    assert {trial.fold_id for trial in trials} == {"test_af", "test_as", "test_bf", "default_bs"}
    assert {trial.method for trial in trials} == {
        "tcn", "bias_only", "drift_only", "bias_drift", "fir_bias_drift_residual"
    }
    assert {trial.seed for trial in trials} == {20260712, 20260713, 20260714}
    assert len({trial.trial_id for trial in trials}) == 60
```

Add failures for a missing fold, duplicate seed, unsupported method, non-30 epoch base config, or a manifest whose test sequence does not match its fold.

- [ ] **Step 2: Run and confirm missing-module failure**

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_penn_matrix_config.py -q
```

Expected: import failure for `penn_matrix_config`.

- [ ] **Step 3: Implement immutable trial specifications**

```python
@dataclass(frozen=True)
class PennTrialSpec:
    trial_id: str
    fold_id: str
    manifest_path: Path
    test_sequence: str
    method: str
    seed: int

def build_trial_specs(config: dict[str, Any], project_root: Path) -> tuple[PennTrialSpec, ...]:
    ...
```

Trial IDs must be exactly `{fold_id}__{method}__seed_{seed}`. Validate all four split manifests with `load_training_split_manifest` before returning specs.

- [ ] **Step 4: Implement method-specific effective configs**

```python
METHOD_OVERRIDES = {
    "tcn": {"model": {"type": "tcn"}, "static": False, "target_mode": "residual"},
    "bias_only": {"model": {"type": "bias_drift_tcn", "use_bias_branch": True, "use_drift_branch": False}, "static": False, "target_mode": "residual"},
    "drift_only": {"model": {"type": "bias_drift_tcn", "use_bias_branch": False, "use_drift_branch": True}, "static": True, "target_mode": "residual"},
    "bias_drift": {"model": {"type": "bias_drift_tcn", "use_bias_branch": True, "use_drift_branch": True}, "static": True, "target_mode": "residual"},
    "fir_bias_drift_residual": {"model": {"type": "bias_drift_tcn", "use_bias_branch": True, "use_drift_branch": True}, "static": True, "target_mode": "competitor_residual"},
}
```

For the hybrid, set `training.competitor_baseline={enabled: true, mode: fir, fir_taps: 9, ridge_alpha: 0.01}`, `zero_init_head: true`, and disable baseline-margin/correction-energy penalties so the ablation changes the target decomposition rather than adding unrelated regularizers.

- [ ] **Step 5: Create the matrix YAML**

The YAML must list the four exact manifests, three seeds, five methods, base model config, 30 epochs, 2048 static context, FIR taps/alpha, output roots, and expected train/validation/test sequences for every fold.

- [ ] **Step 6: Run config tests and Ruff**

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_penn_matrix_config.py -q
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m ruff check src/virtual_imu/training/penn_matrix_config.py tests/test_penn_matrix_config.py
```

Expected: all pass and exactly 60 deterministic specs.

### Task 3: Build Hierarchical Aggregation Before the Long Run

**Files:**
- Create: `src/virtual_imu/eval/penn_matrix_aggregation.py`
- Create: `tests/test_penn_matrix_aggregation.py`

- [ ] **Step 1: Write failing synthetic aggregation tests**

Use unequal sample counts and deliberately different seed/fold errors. Assert that the aggregate first averages seeds per fold and then macro-averages folds, rather than pooling rows or treating 12 trials as independent folds.

```python
overall = aggregate_matrix_trials(trials, baseline_folds)
row = overall.loc[overall.method.eq("bias_drift")].iloc[0]
assert row["gyro_rmse_mean"] == pytest.approx(np.mean(expected_fold_means))
assert row["gyro_rmse_fold_std"] == pytest.approx(np.std(expected_fold_means, ddof=1))
assert row["gyro_rmse_seed_std_mean"] == pytest.approx(np.mean(expected_seed_stds))
assert row["worst_fold"] == "test_bf"
```

- [ ] **Step 2: Run and confirm missing-module failure**

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_penn_matrix_aggregation.py -q
```

- [ ] **Step 3: Implement canonical trial-row extraction**

```python
CORE_METRICS = (
    "gyro_rmse", "gyro_mae", "accel_rmse", "accel_mae",
    "mean_axis_correlation", "normalized_rmse_score",
    "gyro_frequency_band_rmse", "accel_frequency_band_rmse",
    "gyro_coherence", "accel_coherence",
    "gyro_phase_delay_s", "accel_phase_delay_s",
    "imu_preintegration_residual", "delta_rotation_error_rad",
    "delta_velocity_error_m_s", "delta_position_proxy_error_m",
)

def trial_row(fold_id: str, method: str, seed: int | None, metrics: dict[str, Any]) -> dict[str, Any]:
    ...
```

Read signal metrics from `summary`, frequency metrics from `spectral.method`, and navigation metrics from `preintegration.method`.

- [ ] **Step 4: Implement fold and overall tables**

```python
def aggregate_matrix_trials(
    neural_trials: pd.DataFrame,
    analytic_folds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ...
```

Emit `fold_summary` with per-fold seed mean/std and `overall_summary` with fold macro mean/std, average seed std, worst fold, relative improvements versus Identity/FIR/TCN, and rank by macro normalized RMSE.

- [ ] **Step 5: Write CSV/JSON/Markdown artifact helpers and rerun tests**

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_penn_matrix_aggregation.py -q
```

Expected: synthetic hierarchical statistics and rank are exact.

### Task 4: Implement Resumable Baseline and Neural Trial Execution

**Files:**
- Create: `src/virtual_imu/training/penn_matrix_runner.py`
- Create: `tests/test_penn_matrix_runner.py`

- [ ] **Step 1: Write failing smoke/resume tests**

Create four tiny strict manifests and monkeypatch `train_model`/`VirtualIMUGenerator` with deterministic artifacts. Assert:

```python
run_dir = run_penn_matrix(config, smoke=True)
ledger = pd.read_csv(run_dir / "trials.csv")
assert len(ledger) == 60
assert set(ledger.status) == {"complete"}
assert len(list((run_dir / "baselines").glob("*/metrics/*.json"))) == 16

resumed = run_penn_matrix(config, smoke=True, resume_run=run_dir)
assert resumed == run_dir
assert train_model.call_count == 60
```

The second invocation must skip all 60 matching completed trials rather than call training again.

- [ ] **Step 2: Add failing corruption/provenance tests**

Assert that resume refuses a trial when its config hash changed, and final aggregation refuses wrong held-out sequence, non-finite virtual columns, missing checkpoint, or static context sourced outside train sequences.

- [ ] **Step 3: Run and confirm missing-module failure**

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_penn_matrix_runner.py -q
```

- [ ] **Step 4: Implement source run and baseline fold execution**

```python
def run_penn_matrix(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    smoke: bool = False,
    resume_run: Path | None = None,
) -> Path:
    ...
```

Create/snapshot the run before trial 1 and immediately write its absolute path to `outputs/active_penn_matrix_run.txt`. For each fold, load the strict bundle once, fit the four analytic baselines from train only, predict the held-out test once, and write state/predictions/metrics under `baselines/<fold_id>/`.

- [ ] **Step 5: Implement deterministic neural trial execution**

For each `PennTrialSpec`, write the effective config and fingerprint, call `train_model`, restore `best.ckpt`, call `VirtualIMUGenerator.run_frame(test_df)`, evaluate with `evaluate_penn_method`, and write `trial.json` containing fold/method/seed, split sequences, config/manifest hashes, checkpoint, best epoch, static metadata, FIR mode/state for the hybrid, metrics, and status.

- [ ] **Step 6: Implement strict resume checks**

```python
def trial_is_resumable(trial_dir: Path, expected: TrialFingerprint) -> bool:
    payload = json.loads((trial_dir / "trial.json").read_text(encoding="utf-8"))
    return (
        payload["status"] == "complete"
        and payload["fingerprints"] == asdict(expected)
        and Path(payload["checkpoint"]).is_file()
        and Path(payload["prediction"]).is_file()
    )
```

Log every resume decision as `SKIPS`, and fail rather than overwrite a mismatched completed trial.

- [ ] **Step 7: Integrate aggregation and latest pointer**

After all trials pass audit, call the aggregation module, write all tables, provenance, and a lightweight `outputs/latest/`; update `outputs/latest_run.txt` only after successful aggregation.

- [ ] **Step 8: Run runner tests**

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_penn_matrix_runner.py -q
```

Expected: smoke has 60 complete ledger rows; resume performs zero additional training calls; corruption cases fail clearly.

### Task 5: Generate the Chinese Matrix Report from Artifacts

**Files:**
- Create: `src/virtual_imu/eval/penn_matrix_report.py`
- Add tests to: `tests/test_penn_matrix_aggregation.py`

- [ ] **Step 1: Write a failing report-content test**

```python
report = render_penn_matrix_report(overall, folds, trials, provenance)
assert "四折 LOSO" in report
assert "FIR-Residual" in report
assert "最差 fold" in report
assert "预积分" in report
assert "均值 ± fold 标准差" in report
assert "不能等同于跨传感器泛化" in report
```

- [ ] **Step 2: Implement deterministic report rendering**

The report must derive every displayed number from aggregate frames and cover: ranked table, four-way branch ablation, hybrid versus FIR and bias_drift, per-fold rank changes, seed stability, worst fold, spectral/phase behavior, preintegration trade-offs, and limitations.

- [ ] **Step 3: Run aggregation/report tests**

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_penn_matrix_aggregation.py -q
```

Expected: report assertions pass without hard-coded experiment values.

### Task 6: Add CLI, Smoke Mode, and Operational Logging

**Files:**
- Create: `scripts/run_penn_loso_matrix.py`
- Add tests to: `tests/test_penn_matrix_runner.py`

- [ ] **Step 1: Write failing CLI tests**

Assert `--help` documents `--config`, `--smoke`, and `--resume-run`. Assert an invalid resume path exits nonzero.

- [ ] **Step 2: Implement the CLI**

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-run", type=Path)
    args = parser.parse_args()
    print("READS: matrix YAML, four strict split manifests, strict Parquet inputs")
    print("WRITES: one resumable matrix run with trial/fold/overall/report artifacts")
    print("SKIPS: matching completed trials on --resume-run; all raw data mutation")
    print(run_penn_matrix_from_config(args.config, smoke=args.smoke, resume_run=args.resume_run))
```

- [ ] **Step 3: Run CLI help and all focused tests**

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' scripts/run_penn_loso_matrix.py --help
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_system_id_sequence_boundaries.py tests/test_penn_matrix_config.py tests/test_penn_matrix_aggregation.py tests/test_penn_matrix_runner.py -q
```

Expected: help exits 0 and all focused tests pass.

### Task 7: Run a Real Four-Fold One-Epoch Smoke Audit

**Files:**
- Generated: `runs/penn_loso_matrix_3seed_*_smoke/`

- [ ] **Step 1: Run the real smoke matrix**

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\Anaconda3\envs\pinn_imu\python.exe' scripts/run_penn_loso_matrix.py --config configs/penn_loso_matrix_3seed.yaml --smoke
```

Smoke still expands all 60 trials, but overrides each neural trial to one epoch, hidden width 8, one TCN level, static context 16, and CPU. Expected: 60 complete neural trial rows, 16 analytic evaluations, four fold summaries, and one overall table.

- [ ] **Step 2: Audit the smoke artifacts programmatically**

Assert every trial's test prediction contains only its held-out sequence, all six virtual columns are finite, hybrid checkpoints contain `competitor_baseline.mode == "fir"`, drift-consuming checkpoints contain train-derived context only, and `outputs/latest_run.txt` points to this exact smoke run.

- [ ] **Step 3: Fix any smoke-only integration failure and rerun focused tests**

Do not start full training until the complete real-data smoke audit exits 0.

### Task 8: Execute 60 Full GPU Trials and Produce the Final Report

**Files:**
- Generated: one timestamped `penn_loso_matrix_3seed` directory under `runs/`, with its exact path written to `outputs/active_penn_matrix_run.txt`.

- [ ] **Step 1: Start the full matrix**

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\Anaconda3\envs\pinn_imu\python.exe' scripts/run_penn_loso_matrix.py --config configs/penn_loso_matrix_3seed.yaml
```

Expected: CUDA is selected; each trial runs at most 30 epochs and writes its best validation checkpoint.

- [ ] **Step 2: Resume after interruption using the printed source run**

```powershell
$run = (Get-Content outputs/active_penn_matrix_run.txt -Encoding utf8).Trim()
& 'D:\Anaconda3\envs\pinn_imu\python.exe' scripts/run_penn_loso_matrix.py --config configs/penn_loso_matrix_3seed.yaml --resume-run $run
```

Use this only if interrupted. Expected: completed matching trials log `SKIPS`; execution continues at the first absent trial.

- [ ] **Step 3: Audit acceptance counts and provenance**

Require exactly 60 complete neural trials, 16 fold-baseline metric files, four folds, five neural methods, three paired seeds, held-out-only finite predictions, and train-only FIR/context provenance.

- [ ] **Step 4: Run fresh repository verification**

```powershell
$env:PYTHONPATH='src'
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest -q
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m compileall -q src scripts tests
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m ruff check src/virtual_imu/eval/baselines.py src/virtual_imu/training/penn_matrix_config.py src/virtual_imu/training/penn_matrix_runner.py src/virtual_imu/eval/penn_matrix_aggregation.py src/virtual_imu/eval/penn_matrix_report.py scripts/run_penn_loso_matrix.py tests/test_system_id_sequence_boundaries.py tests/test_penn_matrix_config.py tests/test_penn_matrix_aggregation.py tests/test_penn_matrix_runner.py
```

Expected: full tests and compileall pass; touched files have no Ruff errors. Report any unrelated pre-existing whole-repository Ruff warning separately.

- [ ] **Step 5: Inspect and relay the final artifact set**

Read `outputs/latest_run.txt`, then report the exact source run, overall ranked table, four-fold ablation effects, hybrid-versus-FIR gap, seed/fold variability, worst fold, preintegration findings, and links to `report.md`, `overall_summary.csv`, `fold_summary.csv`, `trials.csv`, and `provenance.json`.

## Completion Checklist

- [ ] Four exact strict manifests validated before training.
- [ ] Shared FIR/ARX history resets by sequence during fit and inference.
- [ ] Five neural methods differ only by approved branches/target decomposition.
- [ ] Three paired seeds are present for every fold/method pair.
- [ ] Frozen FIR is train-only, mode `fir`, serialized, and never gradient-updated.
- [ ] Static context is train-only and serialized for drift-consuming methods.
- [ ] Resume is fingerprint-safe and never silently overwrites mismatched trials.
- [ ] Aggregation is seed-within-fold then fold-macro, with worst fold and two variance levels.
- [ ] Final report is generated from artifacts and states the cross-sequence, not cross-sensor, scope.
- [ ] No raw/processed data, checkpoints, predictions, or runs are committed.
