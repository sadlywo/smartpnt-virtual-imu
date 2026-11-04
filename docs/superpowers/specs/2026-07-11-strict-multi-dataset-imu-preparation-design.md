# Strict Multi-Dataset IMU Preparation Design

## Goal

Convert the downloaded PennCOSYVIO, FusionPortable, and NavINST pilot data into physically calibrated low-grade to high-grade IMU pairs. Only pairs that pass spatial, temporal, and signal-quality gates may enter supervised training. A separate diagnostic view preserves enough pre-calibration information to explain failures without allowing uncalibrated samples into training.

## Scope

The first implementation prepares:

| Dataset | Low-grade input | High-grade target | Initial eligibility |
|---|---|---|---|
| PennCOSYVIO | Tango Top accelerometer and gyroscope | VI Sensor ADIS16488 | Strict candidate |
| FusionPortable | Ouster ICM20948 | STIM300 | Strict candidate |
| FusionPortable | Left DAVIS IMU | STIM300 | Strict candidate |
| FusionPortable | Right DAVIS IMU | STIM300 | Strict candidate |
| NavINST | Xsens MTi-670G | KVH1750 | Diagnostic until official transform is available |
| NavINST | Livox HAP internal IMU | KVH1750 | Diagnostic until official transform is available |
| NavINST | Left and right ZED X IMUs | KVH1750 | Diagnostic until official transforms are available |

Camera, LiDAR point-cloud, event, pose, and GNSS/INS processing are outside this preparation phase. NavINST postprocessed `reference.bag` is retained as provenance but is not a raw six-axis target; `/novatel/imu` in `imus.bag` is the KVH1750 target for the selected outdoor trajectory.

## Verified Source Facts

### PennCOSYVIO

The four shared sequences are `af`, `as`, `bf`, and `bs`. Tango Top stores independent timestamped accelerometer and gyroscope streams. VI Sensor `imu.txt` stores timestamp, acceleration XYZ in `m/s^2`, and angular velocity XYZ in `rad/s`. Nominal rates are 128 Hz for Tango acceleration, 100 Hz for Tango angular velocity, and 200 Hz for the ADIS16488.

The official body extrinsics for Tango Top IMU and VI Sensor IMU are published at:

- https://daniilidis-group.github.io/penncosyvio/extrinsic_calib/
- https://daniilidis-group.github.io/penncosyvio/file_format/

The matrices must be copied into a versioned calibration YAML with their published notation and source URL. Their direction convention must be verified by gravity and dynamic-correlation checks before any pair becomes training eligible.

### FusionPortable

The bags contain these synchronized IMU topics:

- `/stim300/imu/data_raw`: STIM300, 200 Hz, tactical-grade target.
- `/os_cloud_node/imu/data_raw`: Ouster-integrated ICM20948, 100 Hz.
- `/stereo/davis_left/imu/data_raw`: left DAVIS internal IMU, approximately 1000 Hz.
- `/stereo/davis_right/imu/data_raw`: right DAVIS internal IMU, approximately 1000 Hz.

The paper calls the DAVIS IMU `MPU6150` in its sensor table and `MPU6050` in prose. The implementation records the canonical project label `DAVIS_MPU6x50` and preserves both upstream names in calibration metadata instead of claiming an unsupported exact part number.

The official calibration chain is rooted at `calibration_files/20220209_calib/calib/` in the FusionPortable Drive folder. Required files are:

- `body_imu.yaml`
- `event_cam00.yaml`, `event_cam00_imu.yaml`
- `event_cam01.yaml`, `event_cam01_imu.yaml`
- `ouster00.yaml`, `ouster00_imu.yaml`

The source paper and calibration references are:

- https://fusionportable.github.io/files/jiao2022fusionportable.pdf
- https://fusionportable.github.io/dataset/fusionportable/

The event-camera low-IMU rotation is composed as:

```text
R_body_from_ecimu = R_body_from_sensor * transpose(R_ecimu_from_sensor)
```

The Ouster low-IMU rotation is composed as:

```text
R_body_from_osimu = R_body_from_sensor * transpose(R_osimu_from_sensor)
```

Quaternion ordering is `(qw, qx, qy, qz)`. Every composed rotation must pass the matrix checks in the quality-gate section and must improve gravity/dynamic agreement over the raw frame.

### NavINST

The selected `imus.bag` contains `/novatel/imu` at approximately 200 Hz, `/xsens/imu` at approximately 100 Hz, `/livox/imu` at approximately 200 Hz, and both ZED X IMUs at approximately 100 Hz. The official platform page identifies `/novatel/imu` as KVH1750 for outdoor trajectories and states that Livox linear acceleration is expressed in units of gravity, so it must be multiplied by `9.8`.

The platform documentation states that `CalibrationData/setup_transformation.txt` provides transforms with KVH1750 as origin:

- https://navinst.github.io/platform/
- https://navinst.github.io/trajectories/

Until that file is downloaded, checksummed, parsed, and validated, all NavINST pair outputs remain diagnostic and `training_eligible=false`.

## Storage And Provenance

Additional calibration files are downloaded under the existing raw roots:

```text
data/raw/<dataset>_pilot/
|-- 00_archives/
|-- 01_extracted/
|-- 02_calibration/       # Original upstream calibration files, unchanged
|-- _manifests/
`-- README.md
```

Prepared outputs use:

```text
data/processed/multi_imu_strict/
|-- penncosyvio/<pair_id>/<sequence_id>.parquet
|-- fusionportable/<pair_id>/<sequence_id>.parquet
|-- splits/<split_id>.json
`-- manifest.json

data/processed/multi_imu_diagnostics/
|-- <dataset>/<pair_id>/<sequence_id>_sample.parquet
|-- <dataset>/<pair_id>/<sequence_id>_quality.json
`-- manifest.json
```

The preparation config is `configs/multi_dataset_strict_prepare.yaml`. It pins source paths, topics, units, calibration files, target rates, filters, quality thresholds, diagnostic sampling, and seed. Every output manifest records config SHA256, input SHA256, calibration SHA256, Git commit, preparation time, row counts, time range, and quality-gate decisions.

## Canonical Strict Schema

The first 20 columns remain compatible with the existing paired-IMU training code:

```text
timestamp_s
gx_low, gy_low, gz_low
ax_low, ay_low, az_low
gx_high, gy_high, gz_high
ax_high, ay_high, az_high
temperature_c
motion_label
sequence_id
source_kind
low_sensor
high_sensor
source_file
```

The strict adapter adds:

```text
dataset_id
pair_id
calibration_id
split_group
alignment_method
lever_arm_compensated
match_dt_ms
quality_flags
training_eligible
```

All gyroscope columns are `rad/s`; all accelerometer columns are `m/s^2`; timestamps are seconds on the source hardware/common clock. Missing temperature is null, never fabricated as 25 degrees Celsius. `training_eligible` is true only when the complete output file passes every blocking gate.

## Spatial Calibration

Both low and high measurements are rotated into the common rig body frame. A quaternion is normalized before conversion, but a norm error greater than `1e-3` blocks the pair rather than silently repairing it.

Gyroscope measurements require rotation only. Accelerometer measurements also require rigid-body lever-arm compensation. For sensor position `r_BS` expressed from body origin to sensor in body coordinates:

```text
f_B = R_BS * f_S
      - alpha_B cross r_BS
      - omega_B cross (omega_B cross r_BS)
```

Each sensor is compensated using only its own transformed gyroscope, never the high-grade target gyroscope for the low-grade input. `omega_B` comes from a causal low-pass filter over that sensor's gyroscope. `alpha_B` comes from a causal endpoint Savitzky-Golay derivative whose window uses current and past samples only. Filter delay is recorded and handled by timestamp alignment. This prevents target leakage and keeps the low-side transform reproducible at inference. The implementation stores both rotation-only and lever-arm-compensated results in diagnostics. If translation is unavailable, dynamic acceleration cannot enter the strict dataset; static intervals may be reported diagnostically but do not bypass this rule.

## Time Alignment And Resampling

Source/header timestamps are authoritative. FusionPortable bag samples already show sub-millisecond to few-millisecond initial header alignment and were hardware synchronized. The approximately `-0.137 s` values present in some calibration YAML files are recorded as upstream metadata but are not blindly applied to the released bag timestamps.

The pipeline performs these steps:

1. Remove exact duplicate timestamps while recording their count.
2. Reject unresolved decreasing timestamps.
3. Compute overlap using original timestamps.
4. Estimate residual lag from transformed gyroscope signals for diagnostics.
5. Apply a constant lag only when explicitly enabled by calibration config and when it improves held-out correlation; automatic per-split lag fitting is forbidden.
6. Apply a causal anti-alias filter before lowering a stream's rate and record its group delay.
7. Build a uniform grid no faster than the lower useful rate of the pair.
8. Interpolate only across gaps no larger than three source median periods.

Target rates are:

| Pair | Target rate |
|---|---:|
| Penn Tango Top to ADIS16488 | 100 Hz |
| Fusion ICM20948 to STIM300 | 100 Hz |
| Fusion left DAVIS IMU to STIM300 | 200 Hz |
| Fusion right DAVIS IMU to STIM300 | 200 Hz |
| NavINST Xsens/ZED to KVH1750 | 100 Hz after calibration is available |
| NavINST Livox to KVH1750 | 200 Hz after calibration is available |

The low-rate signal is never upsampled merely to imitate the high-grade sensor rate. Causal anti-alias filtering, group-delay handling, and resampling parameters are saved in the manifest. A filter may use only samples available at or before its delayed output timestamp.

## Diagnostic View

Diagnostic Parquet contains, on the same sampled timeline:

- raw low/high sensor-frame six-axis values;
- unit-normalized values;
- rotation-only body-frame values;
- lever-arm-compensated values when available;
- low-minus-high residuals at each stage;
- source timestamp distances and interpolation flags;
- the exact gate failure/warning flags.

Diagnostic outputs are deterministically sampled to at most 200,000 rows per pair/sequence. Full-resolution aggregate statistics are written to JSON. Raw source files remain the authoritative full-resolution diagnostic record. Training loaders reject diagnostic roots and reject any row group whose manifest is not training eligible.

## Quality Gates

### Blocking

A strict output is blocked when any condition holds:

- required calibration or translation is absent;
- quaternion norm error exceeds `1e-3`;
- `||R^T R - I||_F >= 1e-5` or `det(R) <= 0.999`;
- unresolved decreasing timestamps or non-finite six-axis values remain;
- temporal overlap is below 90 percent;
- p99 source-to-grid match distance exceeds one target period;
- interpolation would cross a gap above three source median periods;
- transformed active-motion gyroscope correlation is below 0.70;
- residual cross-correlation lag exceeds one target period;
- output row count is below the configured minimum training-window requirement.

### Warning

Warnings do not block by themselves:

- overlap is between 90 and 95 percent;
- active-motion gyroscope correlation is between 0.70 and 0.90;
- static median gravity-norm disagreement exceeds `0.3 m/s^2`;
- more than 0.1 percent of rows are removed as isolated outliers;
- calibration metadata contains an upstream sensor-name inconsistency.

The quality report includes raw versus transformed correlations. A transform that does not improve physically comparable axes is blocked even if its matrix is numerically valid.

## Split Contract

### PennCOSYVIO

Generate leave-one-sequence-out folds over `af`, `as`, `bf`, and `bs`. The default reporting fold uses `af` and `as` for training, `bf` for validation, and `bs` for testing. No window from one sequence may appear in another split.

### FusionPortable

The 3 h 20 min static sequence is calibration/training-only and never appears in a test split. The single MCR dynamic sequence uses contiguous 60/20/20 percent temporal blocks with a 10 second guard gap at both boundaries. All three Fusion sensor pairs share identical boundaries. This split is labelled `within_sequence_temporal` and must not be reported as cross-trajectory generalization.

### NavINST

No strict split is emitted until calibration is complete. Diagnostic outputs preserve trajectory and sensor identity. When enabled, all low-sensor pairs from the same trajectory share a group and cannot be split independently.

### Cross-Dataset Evaluation

Cross-dataset experiments are separate manifests, such as Penn training and Fusion testing. They never replace within-dataset reporting and must preserve dataset, platform, sensor, and sequence group boundaries.

## Components

The implementation is separated into focused modules:

- calibration acquisition and manifesting;
- Penn text stream reader;
- ROS bag IMU stream reader;
- quaternion/extrinsic composition;
- rigid-body lever-arm compensation;
- anti-alias resampling and timestamp matching;
- quality metrics and gate evaluation;
- strict/diagnostic Parquet writer;
- group-aware split generator;
- CLI and human-readable preparation report.

Every CLI invocation logs what it reads, writes, skips, warns about, and blocks. Existing raw data is never modified.

## Testing And Acceptance

Unit tests use small synthetic text streams and bags. They cover unit conversion, quaternion direction, transform composition, lever-arm equations, anti-alias behavior, gap rejection, residual-lag detection, deterministic diagnostics, gate thresholds, schema compatibility, and split leakage.

The small-scale preparation is accepted when:

1. Penn and all three Fusion pairs produce strict candidate files or explicit evidence-based blocked records.
2. NavINST produces diagnostic records and remains blocked until official transforms are present.
3. Every strict output passes schema, finite-value, transform, time, and correlation gates.
4. Repeated preparation is idempotent and skips outputs whose complete provenance fingerprint matches.
5. Split manifests contain no overlapping sequence/time groups and include guard gaps.
6. A report explains row counts, rates, overlap, lag, correlations, warnings, and blocked reasons per pair.
7. All processed data remains ignored by Git and is written only under the E-drive project workspace.

## Follow-Up Boundary

This phase prepares trustworthy paired signals and splits. It does not launch model training. Once accepted, the next phase runs the existing baselines and bias/drift network on the new strict manifests, first within each dataset and then with leave-dataset-out evaluation.
