# ALiMU Leave-Sensor Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a resumable 15-trial ALiMU `leave_sensor_out` matrix for held-out MPU9250, with four train-fitted analytic baselines, seed-aware aggregation, and a Chinese report.

**Architecture:** Keep Penn's four-fold implementation unchanged. Add ALiMU-specific config expansion, aggregation, report, and runner modules that reuse the existing trainer, generator, baseline fitter, metrics evaluator, parquet writer, and artifact hashing conventions. ALiMU uses legacy train/val Parquet paths, so its provenance validates the split manifest and all three input Parquet files instead of Penn's strict split-manifest bundle.

**Tech Stack:** Python, PyTorch, pandas/pyarrow, NumPy, PyYAML, argparse, pytest, Ruff; `D:\Anaconda3\envs\pinn_imu\python.exe` with `PYTHONPATH=src`.

---

## Approved Protocol Amendment: Train-Global Static Prior

The held-out MPU9250 split cannot provide a sensor-keyed static context without leakage. For ALiMU only, replace the formerly planned `reference_scope: train_sensor` with `reference_scope: train_global`. `build_static_context_windows` must select one deterministic 2048-sample low-motion window from the union of training BMX160/LSM9DS1 rows and write it under `__train_global__`. `IMUWindowDataset` and `VirtualIMUGenerator` must resolve every static-context request to that key when this scope is configured. Metadata must identify the selected training sensor and sequence and be auditable by the runner; fallback construction from validation/test input is forbidden. Penn's existing `train_sensor` behavior remains unchanged.

Before this production change, add failing tests that prove (1) a train-global window is selected only from the supplied training rows, (2) unseen MPU9250 datasets resolve the saved global key without building a target window, and (3) configurations that retain `train_sensor` are rejected for the ALiMU matrix. Update the two ALiMU YAML contracts, config-validator constants, and the real-data integration assertion to `train_global`.

## File Responsibilities

- `configs/alimu_leave_sensor_matrix_3seed.yaml`: immutable method/seed/budget/output contract.
- `configs/alimu_leave_sensor_bias_drift_base.yaml`: shared 30-epoch ALiMU base model configuration.
- `src/virtual_imu/training/alimu_matrix_config.py`: validate the held-out sensor split and construct 15 effective trial configs.
- `src/virtual_imu/eval/alimu_matrix_aggregation.py`: validate one-holdout axes and aggregate neural methods across seeds without fabricating fold variance.
- `src/virtual_imu/eval/alimu_matrix_report.py`: generate the held-out-MPU9250 Chinese report from tables/provenance.
- `src/virtual_imu/training/alimu_matrix_runner.py`: create/resume source runs, fit cached baselines, execute/audit trials, and publish aggregates.
- `scripts/run_alimu_leave_sensor_matrix.py`: logged CLI for smoke, full, and resume invocation.
- `docs/ALIMU_MATRIX_RUN_GUIDE.md`: manual full-training and recovery procedure.
- `tests/test_alimu_matrix_config.py`, `tests/test_alimu_matrix_aggregation.py`, `tests/test_alimu_matrix_runner.py`: regression coverage for the new protocol.

No Git commit is created: the user asked to preserve the existing dirty workspace and never commit run/data artifacts.

### Task 1: Define the ALiMU 15-Trial Contract

**Files:**
- Create: `configs/alimu_leave_sensor_bias_drift_base.yaml`
- Create: `configs/alimu_leave_sensor_matrix_3seed.yaml`
- Create: `src/virtual_imu/training/alimu_matrix_config.py`
- Create: `tests/test_alimu_matrix_config.py`

- [ ] **Step 1: Write failing config-expansion tests**

```python
def test_alimu_matrix_expands_to_fifteen_heldout_mpu9250_trials():
    trials = build_alimu_trial_specs(load_config(MATRIX), Path.cwd())
    assert len(trials) == 15
    assert {trial.method for trial in trials} == set(ALIMU_NEURAL_METHODS)
    assert {trial.seed for trial in trials} == {20260712, 20260713, 20260714}
    assert {trial.holdout_sensor for trial in trials} == {"MPU9250"}
    assert len({trial.trial_id for trial in trials}) == 15
```

Add rejection tests for non-30 epochs, missing `temperature_c`, non-disjoint sensor columns, a train/validation frame containing MPU9250, a test frame containing a sensor other than MPU9250, unsupported methods, altered FIR parameters, and duplicate seeds.

- [ ] **Step 2: Run the new test and confirm it fails because the module does not exist**

```powershell
$env:PYTHONPATH='src'
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_alimu_matrix_config.py -q
```

- [ ] **Step 3: Implement immutable trial specifications and validation**

```python
ALIMU_NEURAL_METHODS = (
    "tcn", "bias_only", "drift_only", "bias_drift", "fir_bias_drift_residual",
)
ALIMU_SEEDS = (20260712, 20260713, 20260714)

@dataclass(frozen=True)
class ALiMUTrialSpec:
    trial_id: str
    method: str
    seed: int
    holdout_sensor: str

def build_alimu_trial_specs(config: dict[str, Any], project_root: Path) -> tuple[ALiMUTrialSpec, ...]:
    validate_alimu_matrix(config, project_root)
    return tuple(
        ALiMUTrialSpec(f"mpu9250__{method}__seed_{seed}", method, seed, "MPU9250")
        for method in ALIMU_NEURAL_METHODS
        for seed in ALIMU_SEEDS
    )
```

Validation loads train/val/test Parquet metadata, checks required six-axis target/input columns plus `timestamp_s`, `sequence_id`, `low_sensor`, `temperature_c`, and enforces `BMX160/LSM9DS1` only in train/validation and `MPU9250` only in test.

- [ ] **Step 4: Implement controlled ablation config generation**

```python
def build_alimu_effective_trial_config(base: dict[str, Any], trial: ALiMUTrialSpec, matrix: dict[str, Any]) -> dict[str, Any]:
    effective = deepcopy(base)
    effective.update(seed=trial.seed, experiment_name=trial.trial_id)
    if trial.method == "tcn":
        effective["model"] = {**effective["model"], "type": "tcn", "target_mode": "residual"}
        effective["data"].pop("static_context", None)
    elif trial.method == "bias_only":
        effective["model"].update(type="bias_drift_tcn", target_mode="residual", use_bias_branch=True, use_drift_branch=False)
        effective["data"].pop("static_context", None)
    else:
        effective["model"].update(type="bias_drift_tcn", use_bias_branch=trial.method != "drift_only", use_drift_branch=True)
        effective["data"]["static_context"] = {"enabled": True, "window_size": 2048, "selection": "auto_low_motion", "reference_scope": "train_global"}
        if trial.method == "fir_bias_drift_residual":
            effective["model"].update(target_mode="competitor_residual", zero_init_head=True)
            effective["training"]["competitor_baseline"] = {"enabled": True, "mode": "fir", "fir_taps": 9, "ridge_alpha": 0.01}
    return effective
```

Use ALiMU's seven model input channels (six low IMU axes plus temperature), `window_size: 128`, 30 epochs, the fixed 2048 train-only static context, disabled bias auxiliary loss, and FIR `{taps: 9, ridge_alpha: 0.01}`. The hybrid must set `competitor_baseline.mode: fir`, not `blend`.

- [ ] **Step 5: Create the two YAML files and rerun tests**

The matrix YAML names the three strict Parquet files, source split manifest, holdout `MPU9250`, five methods, three seeds, shared contract, FIR values, and output pointer paths. The base YAML contains the shared optimizer/loss/checkpoint policy and no model-specific residual competitor.

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_alimu_matrix_config.py -q
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m ruff check src/virtual_imu/training/alimu_matrix_config.py tests/test_alimu_matrix_config.py
```

Expected: all pass and exactly 15 deterministic trial IDs are emitted.

### Task 2: Build Single-Holdout Seed Aggregation

**Files:**
- Create: `src/virtual_imu/eval/alimu_matrix_aggregation.py`
- Create: `tests/test_alimu_matrix_aggregation.py`

- [ ] **Step 1: Write failing aggregation tests**

```python
def test_single_holdout_aggregation_means_and_stds_over_three_seeds():
    overall = aggregate_alimu_trials(neural_trials(), analytic_trials())
    hybrid = overall.loc[overall.method.eq("fir_bias_drift_residual")].iloc[0]
    assert hybrid["trial_count"] == 3
    assert hybrid["gyro_rmse_mean"] == pytest.approx(np.mean([0.1, 0.2, 0.3]))
    assert hybrid["gyro_rmse_seed_std"] == pytest.approx(np.std([0.1, 0.2, 0.3], ddof=1))
    assert pd.isna(hybrid["gyro_rmse_fold_std"])
```

Add tests for missing/duplicate seed axes, analytic methods carrying a seed, mismatched sample counts, non-finite core metrics, and rank/reduction against the single identity and FIR baselines.

- [ ] **Step 2: Run and confirm the missing-module failure**

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_alimu_matrix_aggregation.py -q
```

- [ ] **Step 3: Implement canonical flattening and aggregation**

Reuse `trial_row` and `CORE_METRICS` from `penn_matrix_aggregation` to keep metric semantics identical. Implement:

```python
def aggregate_alimu_trials(neural: pd.DataFrame, analytic: pd.DataFrame) -> pd.DataFrame:
    validate_axes(neural, analytic)  # 15 neural + 4 analytic rows
    scored_neural, scored_analytic = score_against_identity(neural, analytic)
    return summarize_per_method(scored_neural, scored_analytic)
```

The summary must contain mean/std across the three neural seeds, one deterministic analytic value, normalized RMSE, rank, relative improvement versus Identity/FIR/TCN, and `holdout_sensor="MPU9250"`. It must not create a `fold_summary.csv` or compute a fold standard deviation.

- [ ] **Step 4: Write CSV/JSON/Markdown tables and verify**

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_alimu_matrix_aggregation.py -q
```

Expected: rank and seed uncertainty are deterministic regardless of incoming row order.

### Task 3: Implement the ALiMU-Specific Report

**Files:**
- Create: `src/virtual_imu/eval/alimu_matrix_report.py`
- Modify: `tests/test_alimu_matrix_aggregation.py`

- [ ] **Step 1: Write a failing report test**

```python
report = render_alimu_matrix_report(overall, trials, provenance)
assert "MPU9250" in report
assert "leave_sensor_out" in report
assert "15" in report
assert "种子标准差" in report
assert "不能与 Penn 的绝对 RMSE 直接比较" in report
```

- [ ] **Step 2: Implement deterministic rendering**

The report renders protocol/provenance, full ranking with `mean ± seed std`, four-way ablation, hybrid-minus-FIR and hybrid-minus-bias-drift differences, seed stability, spectral and preintegration winners, and limitations. It must state that only train/validation use BMX160/LSM9DS1 and all tests use MPU9250/VN100 at 40 Hz.

- [ ] **Step 3: Run report/aggregation tests**

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_alimu_matrix_aggregation.py -q
```

Expected: report content depends only on passed artifacts and never claims cross-dataset transfer.

### Task 4: Implement Resumable Baseline and Neural Execution

**Files:**
- Create: `src/virtual_imu/training/alimu_matrix_runner.py`
- Create: `tests/test_alimu_matrix_runner.py`

- [ ] **Step 1: Write a failing smoke/resume test with tiny ALiMU frames**

```python
run_dir = run_alimu_matrix(config, smoke=True)
ledger = pd.read_csv(run_dir / "trial_ledger.csv")
trials = pd.read_csv(run_dir / "trials.csv")
assert len(ledger) == 15
assert set(ledger.status) == {"complete"}
assert len(trials) == 19
assert len(train_calls) == 15
assert len(baseline_fit_calls) == 1

assert run_alimu_matrix(config, smoke=True, resume_run=run_dir) == run_dir
assert len(train_calls) == 15
assert len(baseline_fit_calls) == 1
```

Use real baseline functions with monkeypatched `train_model` and `VirtualIMUGenerator`. Test fixture frames must retain `low_sensor`, `sequence_id`, and finite six-axis columns.

- [ ] **Step 2: Add failing leakage and corruption tests**

Cover changed base/matrix/split/Parquet fingerprints; a completed baseline state hash change; wrong-sensor test prediction; non-finite virtual axes; static context sourced from a test sequence; missing checkpoint; and a hybrid checkpoint lacking a train-fitted `mode=fir` state.

- [ ] **Step 3: Run and confirm the missing-module failure**

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_alimu_matrix_runner.py -q
```

- [ ] **Step 4: Implement run creation, baseline cache, and fingerprints**

```python
def run_alimu_matrix(config: dict[str, Any], *, config_path: Path | None = None, smoke: bool = False, resume_run: Path | None = None) -> Path:
    specs = build_alimu_trial_specs(config, project_root_from_file(__file__))
    frames = load_and_validate_alimu_splits(config)
    # Snapshot config + split manifest + input parquet hashes before any fit.
    # Write outputs/active_alimu_matrix_run.txt before baseline work.
    # Fit four analytic baselines on frames["train"] only and predict frames["test"].
```

Use `fit_penn_baselines`, `predict_penn_baselines`, `_prediction_frame`, and `evaluate_penn_method`: these functions operate on the canonical columns and do not assume Penn sequence names. Write one `baselines/baseline.json` with `fit_split: train`, allowed fit sensors, held-out sensor, state hash, four prediction/metric hashes, and source fingerprints.

- [ ] **Step 5: Implement neural trial execution and audit**

For each effective config, set an isolated `runs/.../trials/<trial_id>/training` directory, call `train_model`, inspect `best.ckpt`, run only the test frame through `VirtualIMUGenerator`, and write `trial.json`. `audit_prediction` must require `set(frame.low_sensor) == {"MPU9250"}` and six finite virtual columns. `audit_checkpoint` must require static metadata sources to be training sequence IDs and verify hybrid FIR state by refitting it only from the stored train frame.

- [ ] **Step 6: Implement safe resume and publishing**

Compare canonical JSON fingerprints of the matrix/base config, split manifest, and all input Parquet contents before reading cached work. Matching completed baseline/trial artifacts are audited then logged as `SKIPS`; mismatches fail before any training. Write `outputs/latest_alimu_matrix_run.txt` and optional `outputs/latest_alimu_matrix/` only after 15 neural and 4 analytic rows aggregate successfully.

- [ ] **Step 7: Run runner tests**

```powershell
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_alimu_matrix_runner.py -q
```

Expected: 15 complete trials, four cached baseline metrics, zero extra fitting on resume, and all corruption cases fail before overwrite.

### Task 5: Add CLI and Manual Run Guide

**Files:**
- Create: `scripts/run_alimu_leave_sensor_matrix.py`
- Create: `docs/ALIMU_MATRIX_RUN_GUIDE.md`
- Modify: `tests/test_alimu_matrix_runner.py`

- [ ] **Step 1: Write failing CLI help coverage**

```python
result = subprocess.run([sys.executable, "scripts/run_alimu_leave_sensor_matrix.py", "--help"], capture_output=True, text=True)
assert result.returncode == 0
assert "--config" in result.stdout
assert "--smoke" in result.stdout
assert "--resume-run" in result.stdout
```

- [ ] **Step 2: Implement the logged CLI**

```python
parser.add_argument("--config", type=Path, default=Path("configs/alimu_leave_sensor_matrix_3seed.yaml"))
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--resume-run", type=Path)
logger.info("READS ALiMU split manifest and train/validation/test Parquet inputs")
logger.info("WRITES one resumable ALiMU matrix run")
logger.info("SKIPS raw data mutation and matching completed work on resume")
```

- [ ] **Step 3: Write the user-run commands**

The guide documents config inspection, optional smoke, formal execution, resume from `outputs/active_alimu_matrix_run.txt`, expected 15/4/19 counts, and the fact that `outputs/latest_run.txt` remains Penn-specific while the ALiMU pointer is separate.

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'D:\Anaconda3\envs\pinn_imu\python.exe' scripts/run_alimu_leave_sensor_matrix.py --config configs/alimu_leave_sensor_matrix_3seed.yaml
```

- [ ] **Step 4: Run focused verification**

```powershell
$env:PYTHONPATH='src'
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m pytest tests/test_alimu_matrix_config.py tests/test_alimu_matrix_aggregation.py tests/test_alimu_matrix_runner.py -q
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m compileall -q src scripts tests
& 'D:\Anaconda3\envs\pinn_imu\python.exe' -m ruff check src/virtual_imu/training/alimu_matrix_config.py src/virtual_imu/training/alimu_matrix_runner.py src/virtual_imu/eval/alimu_matrix_aggregation.py src/virtual_imu/eval/alimu_matrix_report.py scripts/run_alimu_leave_sensor_matrix.py tests/test_alimu_matrix_config.py tests/test_alimu_matrix_aggregation.py tests/test_alimu_matrix_runner.py
```

Expected: all focused tests, compilation, and lint pass. Do not start formal training during implementation.

## Completion Checklist

- [ ] Exactly 15 deterministic neural trial IDs and four train-only analytic baselines.
- [ ] Train/validation contain only BMX160/LSM9DS1; test contains only MPU9250.
- [ ] Every neural method has exactly three configured seeds and 30 maximum epochs.
- [ ] Static/FIR/normalization state is train-only and auditable.
- [ ] Resume rejects changed input/config/artifact fingerprints and safely skips valid work.
- [ ] Aggregate reports seed uncertainty only, with no fabricated fold variance.
- [ ] Chinese report states the true leave-low-sensor-out scope and correct Penn comparison boundary.
- [ ] User receives formal and resume commands but executes the 15 full trials themselves.
