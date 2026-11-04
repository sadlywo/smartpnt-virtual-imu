# Unified IMU dataset and remote training preparation

The requested deliverable is a built, audited local training corpus and a portable server handoff, not a training run. Original datasets and existing experiments remain traceable. Existing uncommitted work is part of the source snapshot; generated data is never committed.

## Data contract

Build `data/processed/unified_imu_v2/` from ALiMU real sequences, strict PennCOSYVIO pairs and native 200 Hz SmartPNT IMU plus ECEF navigation truth. FusionPortable is excluded because independent frozen-parameter testing finds no supported 80-row test windows. Store a common 40 Hz view (the native ALiMU rate), SI gyroscope rad/s and acceleration m/s², monotonically increasing relative time, namespaced sequence identity, explicit reference frame, source hashes and preprocessing provenance. Preserve native inputs and record incremental plus inherited filtering latency. Split gaps into segments; never manufacture values across gaps. Fit normalization on training records only. v1 is superseded diagnostic and rejected by final training/export validation.

One catalog contains two task views. `paired_signal` has `gx_low/gy_low/gz_low/ax_low/ay_low/az_low` inputs and corresponding `_high` labels. `navigation` has the same input names plus real position, velocity and attitude labels. No navigation truth is converted into a claimed measured high-grade IMU. Record label availability and mapping semantics explicitly. VN100, ADIS16488 and STIM300 are different reference devices; retain reference identity and report metrics per dataset/device.

## Splits and duplicates

ALiMU's three devices share 11 measured VN100 recordings, so the earlier leave-MPU9250 policy is retained only as diagnostic history. Raw CSV checks also confirm two input/time-prefix reuse families per device: dyn_03 into dyn_04 and stat_10 into stat_11. Archive the six shorter inputs, documenting the small reference gyro differences rather than claiming completely identical paired records. The final ALiMU recording split is dyn_01/dyn_02/dyn_04/dyn_05/dyn_06/stat_11 train, dyn_07 validation, dyn_08/dyn_09 test. Keep all three devices for each recording in the same split, and reject shared measured reference content across splits. This evaluates unseen recordings, not an unseen-device holdout. Keep Penn's af/as train, bf validation, bs test. Use SmartPNT Traj3 train, Traj4 validation, Traj1 test. Verify Traj1/Traj2 equality in measurement content, ignoring sequence metadata; retain original Traj2 files and create an archival duplicate record with hashes and canonical identity. Exclude duplicates from every training and export view. Windows cannot cross sequence/segment boundaries.

## Fusion audit

Reproduce failures before changes. Check source cache extent, official extrinsics/time-shift convention, units, header/record timestamps, finite sample support, FIR transients and excitation. Estimate any data-driven correction on a declared calibration portion and evaluate on a disjoint portion. Static records need static-specific diagnostics; correlation of noise is not a motion agreement test. Do not lower gates merely to admit data. Save the decision and metrics in a distinct run.

## Files and reproducibility

Dataset root: `manifest.json`, `strict_manifest.json`, `label_schema.json`, `config_snapshot.yaml`, `archive/duplicates.json`, `sequences/<task>/<dataset>/<sequence>.parquet`, `splits/paired_default.json`, `splits/navigation_default.json`, `normalization.json`. Manifest paths are project-relative POSIX paths; hashes cover all exported Parquet files. `runs/<build_id>/` contains config snapshot, data manifest, source commit plus dirty-source fingerprint, verification results and report. Generated dataset and run mutually identify each other.

## Server handoff

Produce a deterministic allowlisted source/data archive and checksum inventory; include current source contents, including needed untracked modules, without datasets' camera/LiDAR bags, caches, environments, keys or prior checkpoints. Provide Linux bootstrap, GPU/data preflight, dry-run commands, remote transfer guidance and commands for both task configs. Do not rent a machine, connect remotely or start lengthy training before the user supplies a server. Verify an archive round trip in an isolated directory and run a small local loader/trainer smoke test.

## Acceptance

Every included record has validated SI fields, finite input/required labels, fixed-rate segments and a disjoint split assignment. Paired loaders reject navigation-only or ineligible records. Duplicate exclusion is hash-backed. The server archive verifies after extraction and contains all required source files and data. Fusion's inclusion decision includes independent numerical evidence, or a precise remaining blocker.
