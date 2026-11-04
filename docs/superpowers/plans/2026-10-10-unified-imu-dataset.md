# Unified IMU Dataset Implementation Plan

> **For agentic workers:** Use subagent-driven-development for independent audit/server tasks, with spec and quality review; execute dependent dataset integration in this session. Preserve the existing dirty worktree and do not commit generated files.

**Goal:** Build the user's unified, labelled IMU corpus and a verified portable training handoff.

**Architecture:** Separate the Fusion audit, deterministic dataset preparation, and server packaging. A shared catalog carries two genuine task schemas with compatible paired/navigation views, portable manifests and explicit duplicate exclusion.

**Tech Stack:** Python, NumPy/SciPy, pandas/Arrow Parquet, YAML, PyTorch, standard-library tarfile/hashlib.

## Task 1: Fusion diagnosis

- [x] Reproduce quality failures from cached source streams and inspect official calibration/time conventions.
- [x] Test time/extrinsic hypotheses on a calibration subset, validate independently and save a run report.
- [x] Add regression tests for confirmed bugs before implementing corrections; document exclusions that remain.

## Task 2: Common corpus

Files: new `configs/unified_imu_dataset.yaml`, `src/virtual_imu/io/unified_dataset.py`, `src/virtual_imu/preprocessing/unified_signals.py`, `scripts/prepare_unified_imu.py`, `tests/test_unified_dataset.py`.

- [x] Test SI fields, anti-aliasing, gap segmentation, content deduplication and task-specific labels using small physical signals.
- [x] Run `.venv312/Scripts/python.exe -B -m pytest tests/test_unified_dataset.py` and confirm the missing-function failure.
- [x] Build deterministic 40 Hz segments with matched filtering of paired input/target, source provenance and immutable version identity.
- [x] Generate strict/split manifests, train-only normalization and duplicate archival records; exclude navigation from measured IMU labels.
- [x] Run preparation on all local data, validate exported hashes and real dataloader batches; save source-of-truth run report.

## Task 3: Training and server handoff

Files: new paired/navigation YAMLs, `scripts/package_training_bundle.py`, `scripts/server/`, `docs/SERVER_TRAINING.md`, `tests/test_training_bundle.py`.

- [x] Make both task views consumable by existing trainers with portable paths.
- [x] Test archive allowlisting, checksum verification, tampering rejection and safe extraction before implementation.
- [x] Build the source/data bundle, bootstrap and GPU/data preflight; explicitly document the tested resume support.
- [x] Extract and verify the built bundle independently; smoke-test the paired trainer and navigation labels without a full training run.

## Task 4: Review and deliver

- [x] Review spec coverage, then scientific correctness and code quality; resolve actionable findings.
- [x] Run targeted tests, affected regression suites and Ruff. Record exact passing checks and unresolved limitations.
- [x] Prepare delivery of local corpus path, Fusion decision, mapping table, dataset statistics and server handoff instructions.

## Verified implementation state before packaging

Formal `unified_imu_v2` contains 944,840 rows in 145 contiguous records at 40 Hz. ALiMU is split by physical reference recording; the six shorter reused input variants and SmartPNT Traj2 are archived without deleting their original files. Every possible 80-row LOW/HIGH window was checked with zero exact overlap across splits. Superseded v1 is rejected by formal loaders, export and server preflight.

The two task loaders each completed two finite CUDA optimization steps on real data. Affected regression suites passed 42 tests; server packaging/preflight passed 36 tests, Ruff and Bash syntax checks. This verifies data and training readiness, not model accuracy or the future Linux/GPU runtime.

Final bundle: `outputs/server_bundles/server_bundle_20261010_142710_400018.tar.gz` (123,145,882 bytes); source-of-truth `runs/server_bundle_20261010_142710_400018/`. Isolated extraction verified all 417 file checksums, 145 curated records, task schemas and selected build provenance. The archive SHA256 was independently re-read and matches its sidecar: `79079e11b2c196442148a435d16aecbf896a8aa0ef8376a7eb09f872ecfcae3d`. This post-export plan update is not part of that immutable source snapshot.
