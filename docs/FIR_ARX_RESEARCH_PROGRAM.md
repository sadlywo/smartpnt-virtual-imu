# FIR/ARX Research Program

## Goal

Test whether preprocessing, topological routing, or sparse component selection can make causal FIR/ARX correction outperform strong device-specific calibration baselines on short-window low-to-high IMU signal reproduction.

This is deliberately narrower than a new end-to-end virtual-IMU framework. The target is a measurable device-level gain in the paired-signal task, with unchanged LOW-to-HIGH references and recording-disjoint evaluation.

## Branch layout

- `codex/fir-arx-reproduction`: frozen data, protocol construction, identity/affine/FIR/ARX baselines, evaluation, report, and figure infrastructure.
- `codex/fir-arx-vmd`: VMD decomposition before FIR/ARX; low, high, and combined modes.
- `codex/fir-arx-tda`: persistent-homology descriptors, train-only motion-regime clustering, and cluster-specific FIR/ARX.
- `codex/fir-arx-minimal`: low/high-frequency component candidates, scene-aware selection, and a component-count versus accuracy floor.

Each experiment branch starts from the reproduction branch and shares its data support and metrics.

## Immutable evaluation contract

1. Data comes from `data/processed/unified_imu_v2/manifest.json`; per-record SHA256 values are checked.
2. Every model is fit on `train`; hyperparameters and selected components are chosen on `validation`; `test` is evaluated once.
3. Protocol logic is recording-disjoint. ALiMU low sensors sharing a VN100 recording remain in the same split.
4. Models are fitted separately per `(dataset_id, pair_id, low_sensor, high_sensor)`. Dataset/protocol summaries are unweighted macro averages.
5. FIR features are causal lags of LOW only. ARX-style features add only the current LOW first difference; past HIGH or predicted HIGH is never fed back.
6. Results retain failures and degradations. Seed SD is not interpreted as recording-level uncertainty.

## Primary metrics

- Gyroscope vector RMSE in rad/s.
- Accelerometer vector RMSE in m/s².
- 0.5 Hz low/high-pass RMSE decomposition as a diagnostic for slow bias versus motion-band correction.
- Improvement percentage relative to the uncorrected LOW signal.

The primary decision is made on held-out recording metrics, never on the training loss or the best validation point alone.

## Why FIR/ARX first

The latest public-data study found strong device-specific FIR/ARX controls. A preprocessing idea is credible only if it improves over those controls under the same information scope. The project therefore treats FIR/ARX as the object of study rather than as a weak baseline.
