# Strict Multi-Dataset IMU Preparation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, physically calibrated PennCOSYVIO and FusionPortable low-to-high IMU preparation pipeline, while emitting explicitly non-trainable NavINST diagnostics until its official transform file is available.

**Architecture:** Source adapters first extract independent timestamped IMU streams without assuming alignment. Focused preprocessing modules then apply versioned extrinsics, causal lever-arm compensation, causal anti-alias resampling, and quality gates. The orchestrator writes strict Parquet only for passing pairs, sampled diagnostic Parquet for every pair, provenance manifests, leakage-safe splits, and a readable report.

**Tech Stack:** Python 3.9-compatible syntax in the existing `pinn_imu` environment, NumPy, pandas, SciPy, PyArrow, PyYAML, rosbags, Typer, Loguru, pytest, and Ruff.

No Git commit steps are included because the user did not request commits and the current worktree contains unrelated changes.

---

## File Map

- Create `configs/multi_dataset_strict_prepare.yaml`: authoritative source, calibration, filter, gate, diagnostic, and split settings.
- Create `configs/calibration/penncosyvio_body.yaml`: published Tango Top and VI Sensor body extrinsics with source URLs.
- Create `src/virtual_imu/io/multi_imu_calibration.py`: calibration asset acquisition, OpenCV-YAML parsing, provenance, and pair transform construction.
- Create `src/virtual_imu/io/multi_imu_sources.py`: Penn text parsing and one-pass multi-topic ROS bag extraction.
- Create `src/virtual_imu/preprocessing/rigid_imu.py`: quaternion validation, transform composition, rotation, and causal lever-arm compensation.
- Create `src/virtual_imu/preprocessing/strict_alignment.py`: timestamp cleaning, causal filtering/resampling, match-distance, and lag diagnostics.
- Create `src/virtual_imu/preprocessing/imu_quality.py`: metrics, warnings, blocking gates, and training eligibility.
- Create `src/virtual_imu/io/multi_imu_prepare.py`: pair orchestration, Parquet schema, diagnostic sampling, fingerprints, manifests, and report data.
- Create `src/virtual_imu/io/multi_imu_splits.py`: Penn leave-one-sequence-out and Fusion guarded temporal split manifests.
- Create `scripts/prepare_multi_imu_strict.py`: Typer CLI.
- Create `tests/test_multi_imu_calibration.py`.
- Create `tests/test_multi_imu_sources.py`.
- Create `tests/test_rigid_imu.py`.
- Create `tests/test_strict_alignment.py`.
- Create `tests/test_imu_quality.py`.
- Create `tests/test_multi_imu_prepare.py`.
- Create `tests/test_multi_imu_splits.py`.
- Modify `src/virtual_imu/training/dataset.py`: reject explicit non-trainable frames without changing legacy ALiMU behavior.
- Modify `README.md`: document preparation and source-of-truth manifests.

Generated artifacts remain under `data/raw`, `data/processed`, and `outputs`; none are committed.

### Task 1: Calibration Configuration And Asset Manifest

**Files:**
- Create: `configs/multi_dataset_strict_prepare.yaml`
- Create: `configs/calibration/penncosyvio_body.yaml`
- Create: `src/virtual_imu/io/multi_imu_calibration.py`
- Test: `tests/test_multi_imu_calibration.py`

- [ ] **Step 1: Write failing tests for OpenCV matrices, quaternion composition, and missing NavINST calibration**

```python
def test_compose_event_imu_to_body_rotation():
    sensor_to_body = quaternion_wxyz_to_matrix([1.0, 0.0, 0.0, 0.0])
    sensor_to_local_imu = quaternion_wxyz_to_matrix([0.0, 0.0, 0.0, 1.0])
    result = compose_reference_to_body(sensor_to_body, sensor_to_local_imu)
    np.testing.assert_allclose(result, sensor_to_body @ sensor_to_local_imu.T)


def test_missing_required_navinst_transform_is_diagnostic_only(tmp_path):
    status = calibration_status(tmp_path / "setup_transformation.txt", required=True)
    assert status == {"available": False, "training_eligible": False, "reason": "missing_calibration"}
```

Also test that `!!opencv-matrix` is parsed into the declared `(rows, cols)` shape and that quaternion norm error above `1e-3` raises `ValueError`.

- [ ] **Step 2: Run the tests and confirm missing imports/functions**

Run:

```powershell
$env:PYTHONPATH = "src"
& "D:\Anaconda3\envs\pinn_imu\python.exe" -m pytest tests\test_multi_imu_calibration.py -q
```

Expected: collection failure because `virtual_imu.io.multi_imu_calibration` does not exist.

- [ ] **Step 3: Implement safe calibration parsing and transform construction**

Implement the public interfaces `load_opencv_yaml(path: Path) -> dict[str, Any]`, `opencv_matrix(value: dict[str, Any]) -> np.ndarray`, `calibration_status(path: Path, required: bool) -> dict[str, Any]`, `load_penn_body_transforms(path: Path) -> dict[str, RigidTransform]`, `load_fusion_body_transforms(calibration_root: Path) -> dict[str, RigidTransform]`, and `acquire_calibration_assets(config: dict[str, Any]) -> dict[str, Any]`.

Use a local `yaml.SafeLoader` subclass with a constructor for `tag:yaml.org,2002:opencv-matrix`; do not register a process-global constructor. Reuse `resolve_gdrive_matches`, `_gdrive_direct_download_url`, `download_artifact`, `sha256_file`, and atomic JSON writing patterns from `pilot_download.py`. Write assets to `data/raw/<dataset>_pilot/02_calibration` and `calibration_manifest.json` without overwriting `pilot_manifest.json`.

- [ ] **Step 4: Add complete calibration config**

The YAML must include the seven verified Fusion files, the published Penn matrices and URLs, and this explicit NavINST state:

```yaml
navinst:
  expected_path: data/raw/navinst_pilot/02_calibration/setup_transformation.txt
  source_directory: https://www.frdr-dfdr.ca/repo/dataset/8f8b4d74-3264-4e7d-b851-8a7b4e804cf8
  required_for_strict: true
  missing_behavior: diagnostic_only
```

Do not invent a NavINST file URL. The config records the known official directory and expected filename until the exact path is verified.

- [ ] **Step 5: Run focused tests and Ruff**

Expected: calibration tests pass; malformed matrices, bad quaternions, ambiguous Drive matches, and missing required transforms are explicit errors/statuses.

### Task 2: Source Stream Adapters

**Files:**
- Create: `src/virtual_imu/io/multi_imu_sources.py`
- Test: `tests/test_multi_imu_sources.py`

- [ ] **Step 1: Write failing Penn parsing tests**

Create tiny Tango accel/gyro and VI Sensor text fixtures. Assert exact SI columns and independent timestamps:

```python
accel = read_penn_vector_file(accel_path, value_columns=["ax", "ay", "az"])
gyro = read_penn_vector_file(gyro_path, value_columns=["gx", "gy", "gz"])
high = read_penn_vi_imu(vi_path)
assert accel.columns.tolist() == ["timestamp_s", "ax", "ay", "az"]
assert high.columns.tolist() == ["timestamp_s", "ax", "ay", "az", "gx", "gy", "gz"]
```

Include duplicate and decreasing timestamp fixtures.

- [ ] **Step 2: Write a failing message-conversion and topic-extraction test**

Test `_imu_message_row` with `SimpleNamespace` sensor messages, including header timestamp preference, bag timestamp fallback, frame ID, and Livox acceleration scale `9.8`. Test `extract_rosbag_imu_topics` with an injected message iterator so unit tests do not require a multi-gigabyte bag.

- [ ] **Step 3: Implement source adapters**

Implement `read_penn_vector_file(path: Path, value_columns: list[str]) -> pd.DataFrame`, `read_penn_vi_imu(path: Path) -> pd.DataFrame`, `extract_penn_sequence(root: Path, sequence: str, cache_root: Path) -> dict[str, Path]`, and `extract_rosbag_imu_topics(bag_path: Path, topic_specs: dict[str, dict[str, Any]], cache_root: Path, chunk_rows: int, message_iterator: Callable | None = None) -> dict[str, Any]`.

Scan each bag once for all selected topics. Write one chunked PyArrow Parquet per topic with `timestamp_s`, six SI axes, `frame_id`, `record_timestamp_s`, and `source_file`. Preserve source order, count duplicates/non-monotonic timestamps, and never load an entire Fusion bag into memory.

- [ ] **Step 4: Verify focused tests and a real index-only smoke check**

Run unit tests, then extract at most the first 10 seconds from NavINST and Fusion MCR using a config override. Expected: topic names, rates, frame IDs, and scale statistics match the inspection evidence.

### Task 3: Rigid-Body IMU Calibration Without Target Leakage

**Files:**
- Create: `src/virtual_imu/preprocessing/rigid_imu.py`
- Test: `tests/test_rigid_imu.py`

- [ ] **Step 1: Write failing rotation validation tests**

Test identity, a known 90-degree rotation, quaternion normalization tolerance, determinant, and orthogonality thresholds.

- [ ] **Step 2: Write failing lever-arm physics tests**

For constant `omega=[0, 0, 2]`, zero angular acceleration, and `r=[1, 0, 0]`, assert that shifting a sensor acceleration back to the body origin removes `[-4, 0, 0]`. Test a linearly increasing angular velocity for the tangential term.

- [ ] **Step 3: Write a target-leakage regression test**

```python
parameters = inspect.signature(compensate_lever_arm).parameters
assert list(parameters) == ["frame", "transform", "config"]
low_corrected_a = compensate_lever_arm(low, transform, filter_config)
low_corrected_b = compensate_lever_arm(low.copy(), transform, filter_config)
pd.testing.assert_frame_equal(low_corrected_a, low_corrected_b)
```

The low result must depend only on low samples and fixed calibration.

- [ ] **Step 4: Implement rigid transforms and causal compensation**

Create frozen dataclass `RigidTransform` with `rotation_body_from_sensor: np.ndarray`, `position_body_to_sensor_m: np.ndarray`, and `calibration_id: str`. Implement `quaternion_wxyz_to_matrix(q: Sequence[float]) -> np.ndarray`, `validate_rotation(rotation: np.ndarray) -> None`, `rotate_imu(frame: pd.DataFrame, transform: RigidTransform) -> pd.DataFrame`, `causal_angular_kinematics(gyro: np.ndarray, rate_hz: float, config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]`, and `compensate_lever_arm(frame: pd.DataFrame, transform: RigidTransform, config: dict[str, Any]) -> pd.DataFrame`.

Use `scipy.signal.sosfilt` for causal low-pass filtering and a one-sided endpoint Savitzky-Golay coefficient vector for angular acceleration. Return algorithmic delay. Never accept a high-stream argument when compensating the low stream.

- [ ] **Step 5: Run tests and numerical sanity checks**

Expected: physics tests pass to `1e-8`, no pre-response occurs before a synthetic gyro step, and bad rotations fail before signal processing.

### Task 4: Strict Causal Alignment And Resampling

**Files:**
- Create: `src/virtual_imu/preprocessing/strict_alignment.py`
- Test: `tests/test_strict_alignment.py`

- [ ] **Step 1: Write failing timestamp/gap tests**

Cover exact duplicate removal, decreasing timestamp rejection, overlap computation, and refusal to interpolate across a gap above three median source periods.

- [ ] **Step 2: Write failing causal anti-alias tests**

Use a mixed 2 Hz plus 80 Hz signal sampled at 200 Hz and resampled to 100 Hz. Assert attenuation of the aliased component, preserved 2 Hz amplitude, and no output dependency on input samples later than the declared algorithmic latency.

- [ ] **Step 3: Write failing lag and match-distance tests**

Assert recovery of a known 10 ms gyro-norm shift, p50/p99 nearest-source distance, and a blocked result when residual lag exceeds one target period.

- [ ] **Step 4: Implement strict alignment**

Implement `clean_timestamps(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]`, `causal_resample(frame: pd.DataFrame, grid_s: np.ndarray, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]`, `residual_gyro_lag(low: pd.DataFrame, high: pd.DataFrame, rate_hz: float, max_lag_s: float) -> dict[str, float]`, and `align_strict_pair(low: pd.DataFrame, high: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]`.

Use a causal FIR (`firwin` plus `lfilter`) and record group delay. Permit linear interpolation only with an explicit one-source-period availability latency; otherwise use the latest available filtered sample. Save `match_dt_ms` and interpolation/gap flags per output row. Do not call legacy `align_low_high`, which can fit delay automatically on the evaluation sequence.

- [ ] **Step 5: Run alignment tests and Ruff**

Expected: all causality, delay, gap, and anti-alias tests pass.

### Task 5: Quality Metrics And Training Eligibility

**Files:**
- Create: `src/virtual_imu/preprocessing/imu_quality.py`
- Test: `tests/test_imu_quality.py`

- [ ] **Step 1: Write threshold-boundary tests**

Create reports immediately below, at, and above every configured threshold. Assert blocking for missing calibration, invalid rotations, non-finite values, overlap below 0.90, p99 match distance above one period, gyro correlation below 0.70, excessive residual lag, and insufficient rows.

- [ ] **Step 2: Write warning-only tests**

Assert warning without blocking for overlap `0.90-0.95`, correlation `0.70-0.90`, gravity difference above `0.3 m/s^2`, isolated outliers above 0.1 percent, and the DAVIS MPU6050/6150 naming inconsistency.

- [ ] **Step 3: Implement quality evaluation**

Create frozen dataclass `QualityDecision` with `training_eligible: bool`, `blocking_reasons: tuple[str, ...]`, `warnings: tuple[str, ...]`, and `metrics: dict[str, float | int | str | None]`. Implement `evaluate_pair_quality(raw_metrics: dict[str, Any], transformed_metrics: dict[str, Any], thresholds: dict[str, Any]) -> QualityDecision`.

Require transformed active-axis correlation to improve over raw physically comparable axes. Serialize stable reason codes, not only prose.

- [ ] **Step 4: Run focused tests**

Expected: exact threshold behavior is deterministic and JSON serializable.

### Task 6: Pair Orchestration, Strict Schema, Diagnostics, And Fingerprints

**Files:**
- Create: `src/virtual_imu/io/multi_imu_prepare.py`
- Test: `tests/test_multi_imu_prepare.py`
- Modify: `src/virtual_imu/training/dataset.py`

- [ ] **Step 1: Write a failing end-to-end synthetic pair test**

Prepare a small rotated/delayed low/high pair with known lever arm. Assert the exact 29-column canonical schema, SI units, transformed agreement, `training_eligible=true`, strict and diagnostic paths, and manifest fingerprints.

- [ ] **Step 2: Write failing blocked-pair tests**

Missing NavINST calibration must write diagnostics and a blocked manifest record but no strict Parquet. A low-correlation pair must behave the same way.

- [ ] **Step 3: Write a failing deterministic diagnostic-sampling test**

For 500,000 rows and cap 200,000, assert exactly 200,000 monotonically ordered sampled rows and identical selected indices for the configured seed.

- [ ] **Step 4: Implement the orchestrator**

Define these exact constants:

```python
STRICT_BASE_COLUMNS = [
    "timestamp_s",
    "gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low",
    "gx_high", "gy_high", "gz_high", "ax_high", "ay_high", "az_high",
    "temperature_c", "motion_label", "sequence_id", "source_kind",
    "low_sensor", "high_sensor", "source_file",
]
STRICT_METADATA_COLUMNS = [
    "dataset_id", "pair_id", "calibration_id", "split_group",
    "alignment_method", "lever_arm_compensated", "match_dt_ms",
    "quality_flags", "training_eligible",
]
```

Implement `prepare_pair(pair_config: dict[str, Any], context: PreparationContext) -> dict[str, Any]`, `prepare_multi_dataset(config: dict[str, Any], datasets: list[str] | None = None, dry_run: bool = False) -> dict[str, Any]`, and `provenance_fingerprint(config_path: Path, source_paths: list[Path], calibration_paths: list[Path]) -> str`.

Write strict data under `data/processed/multi_imu_strict/<dataset>/<pair>/<sequence>.parquet`. Write sampled diagnostics and full JSON metrics under `data/processed/multi_imu_diagnostics`. Use atomic manifest writes. Skip only when the complete fingerprint matches and all declared outputs exist.

- [ ] **Step 5: Guard the training loader**

Add to `ensure_training_columns`:

```python
if "training_eligible" in out.columns and not out["training_eligible"].fillna(False).all():
    raise ValueError("Input contains rows that are not eligible for training")
```

Legacy inputs without this column remain supported. Add tests showing eligible strict data loads, diagnostic data fails, and ALiMU behavior is unchanged.

- [ ] **Step 6: Run orchestration and existing training dataset tests**

Run `tests/test_multi_imu_prepare.py`, `tests/test_training_smoke.py`, `tests/test_alimu_reader.py`, and `tests/test_bias_drift_tcn.py`.

### Task 7: Leakage-Safe Split Manifests

**Files:**
- Create: `src/virtual_imu/io/multi_imu_splits.py`
- Test: `tests/test_multi_imu_splits.py`

- [ ] **Step 1: Write failing Penn LOSO tests**

Assert four folds, no sequence overlap, and default fold `train={af,as}`, `val={bf}`, `test={bs}`.

- [ ] **Step 2: Write failing Fusion guarded temporal tests**

Assert 60/20/20 chronological blocks, 10 second gaps, identical boundaries across all three pair IDs, and static sequence exclusion from validation/test.

- [ ] **Step 3: Implement split generation**

Implement `penn_loso_manifests(strict_manifest: dict[str, Any], output_root: Path) -> list[Path]`, `fusion_temporal_manifest(strict_manifest: dict[str, Any], output_root: Path, guard_s: float) -> Path`, and `validate_no_split_leakage(manifest: dict[str, Any]) -> None`.

Split manifests reference immutable strict Parquet plus row/time filters instead of duplicating full datasets. Include `split_kind`, group keys, exact boundaries, guard intervals, and eligibility requirements.

- [ ] **Step 4: Run split tests**

Expected: all group/time overlap attempts raise before a manifest is written.

### Task 8: CLI, Report, And Documentation

**Files:**
- Create: `scripts/prepare_multi_imu_strict.py`
- Modify: `README.md`
- Test: `tests/test_multi_imu_prepare.py`

- [ ] **Step 1: Write a failing CLI help test**

Assert `--config`, repeatable `--dataset`, `--fetch-calibration`, `--dry-run`, and `--max-duration-s` are present under Python 3.9 with UTF-8 output.

- [ ] **Step 2: Implement Typer CLI**

The CLI loads YAML, optionally acquires calibration assets, logs every `READS/WRITES/SKIPS/WARNS/BLOCKS`, supports bounded smoke extraction, and prints the strict manifest path. `--max-duration-s` is an inspection override recorded in provenance and cannot overwrite full-run outputs because it changes the fingerprint/output ID.

- [ ] **Step 3: Generate a readable report**

Write `data/processed/multi_imu_strict/report.md` with one row per pair/sequence: source rates, output rate, rows, duration, overlap, lag, p99 match distance, raw/transformed correlation, gravity disagreement, warnings, blocks, and strict path.

- [ ] **Step 4: Document commands and source-of-truth files**

Add PowerShell examples using `D:\Anaconda3\envs\pinn_imu\python.exe`. Document strict and diagnostic manifests, NavINST's calibration block, and the fact that `within_sequence_temporal` is not cross-trajectory evidence.

- [ ] **Step 5: Run CLI, tests, compile, and Ruff offline**

Expected: CLI help and dry-run succeed without reading entire bags; focused tests and lint pass.

### Task 9: Real Small-Scale Preparation And Validation

**Files:**
- Generated only under `data/raw/*_pilot/02_calibration`, `data/processed/multi_imu_strict`, and `data/processed/multi_imu_diagnostics`.

- [ ] **Step 1: Acquire and checksum Fusion calibration assets**

Run the CLI with `--fetch-calibration --dry-run`, then acquire only the seven declared small files. Expected: all have SHA256 and remain on E drive; NavINST remains explicitly missing.

- [ ] **Step 2: Run a bounded 120-second smoke preparation**

Prepare Penn `af`, Fusion MCR, and Fusion static with `--max-duration-s 120`. Expected: Penn and three Fusion pairs either pass or produce evidence-based blocked records; NavINST diagnostics are emitted with no strict files.

- [ ] **Step 3: Inspect smoke quality before full processing**

Check transform improvement, residual lag, gravity norms, source match distance, and causal filter delay. If a pair blocks, fix only demonstrated parser/convention defects; do not relax thresholds merely to make it pass.

- [ ] **Step 4: Run full Penn preparation**

Expected: `af/as/bf/bs` strict or explicitly blocked per the unchanged quality config, plus four LOSO manifests.

- [ ] **Step 5: Run full Fusion preparation**

Scan each bag once, extract all four topics, prepare three pairs, and write shared guarded temporal boundaries. Static remains training/calibration-only.

- [ ] **Step 6: Run NavINST diagnostics**

Normalize Livox acceleration by 9.8 and report raw-frame timing/rate/value diagnostics for Xsens, Livox, and both ZED IMUs versus KVH1750. Verify no strict NavINST Parquet exists.

- [ ] **Step 7: Repeat the full command**

Expected: matching fingerprints cause verified skips; no bag is rescanned and no Parquet is rewritten.

- [ ] **Step 8: Final verification**

Run focused tests and Ruff again, parse every manifest, confirm every strict row has `training_eligible=true`, validate split leakage, verify no unfinished temporary files, confirm all generated data is ignored by Git, and record remaining E-drive capacity.
