# Project Framework

## Research goal

The project tests whether a low-cost or lower-grade six-axis IMU can be mapped toward a higher-grade six-axis reference, and whether the resulting predictions improve downstream inertial state estimation. It also compares neural sequence models with causal and noncausal signal-processing baselines on the same frozen inputs and evaluation protocol.

The work is structured as a retrospective common-task study. A method name such as DeepGyro, TLIO, TartanIMU, Informer, PatchTST, or iTransformer identifies an adapted backbone or reference family; it does not imply a complete reproduction of every component from the original paper. The accepted report must be read together with the task and information-scope labels.

## Experimental units

### Paired signal mapping

- Input window: 80 samples at 40 Hz, spanning 1.975 seconds.
- Input channels: six LOW gyro/accel axes plus a device-class one-hot vector, for nine channels in the unified protocol.
- Prediction: six normalized HIGH-minus-LOW residual channels.
- Reconstruction: inverse standardization followed by addition of the LOW signal.
- Trainable device mappings: ALiMU BMX160, MPU9250, and LSM9DS1 to VN100; PennCOSYVIO Tango Top IMU to ADIS16488.
- Temperature is retained only where available as metadata; it is not a model input in the frozen v2 protocol. There is no independent static-signal branch.

### SmartPNT navigation exploration

- Input window: 80 samples at 40 Hz.
- Output: three ENU displacement components between the first and last sample of the window.
- Unit: metres for each component and for vector RMSE.
- SmartPNT-MSF has no HIGH six-axis signal label in this task. Overlapping window displacements are never accumulated into a full trajectory, and the study does not claim trajectory ATE or RPE.

## Data and split policy

- ALiMU is split at the physical sensor/recording level with strict train, validation, and test separation.
- PennCOSYVIO uses leave-one-sequence-out families; body-frame calibration is reused only as an explicit calibration input.
- Duplicate or archive-equivalent trajectories are grouped before splitting. SmartPNT `Traj2` must not be treated as independent evidence for `Traj1`.
- FusionPortable was inspected for alignment and signal quality but is not part of the frozen formal paired-training benchmark.
- Test partitions are evaluated only after train/validation model selection is frozen. Hyperparameter values are never selected from the test split.

## Scientific invariants

1. Every experiment is described by a YAML file under `configs/` and a seeded run configuration.
2. Gyro units (`rad/s`) and accelerometer units (`m/s^2`) are reported separately; they are never averaged into a synthetic composite error.
3. Causal/online methods, window methods, and noncausal offline smoothers remain separate comparison groups.
4. Identity, bias/mean, affine, ridge, FIR, ARX, and learned neural baselines are retained as controls.
5. A classical known-device bank and a shared-input neural mapper have different information scopes and must remain distinguishable.
6. Training seeds estimate optimizer variability on one split; sample SD across seeds is not a confidence interval over recordings.
7. The run directory is the source of truth. `outputs/latest/` is a convenience pointer only and must be resolved through its recorded run ID.

## Processing architecture

```mermaid
flowchart LR
    A[Public raw data] --> B[Inspection and provenance manifests]
    B --> C[Alignment and signal preparation]
    C --> D[Unified 40 Hz windows and labels]
    D --> E[Train / validation / test membership]
    E --> F[Baseline and neural training]
    F --> G[Frozen checkpoint selection]
    G --> H[Test evaluation and downstream diagnostics]
    H --> I[Aggregation and statistical summaries]
    I --> J[Nature-style figures and indexed report]
    J --> K[Checksum-verified external result package]
```

The Python package is organized by responsibility:

| Path | Responsibility |
|---|---|
| `src/virtual_imu/io/` | Public dataset readers and file-format boundaries |
| `src/virtual_imu/preprocessing/` | Parsing, synchronization, alignment, calibration, and unified-window construction |
| `src/virtual_imu/models/` | TCN, recurrent, convolution, attention, and published-backbone adapters |
| `src/virtual_imu/training/` | Datasets, trainers, formal pipelines, matrices, HPO, checkpointing, and metrics logging |
| `src/virtual_imu/eval/` | Metrics, baselines, downstream evaluation, ranking, and report inputs |
| `src/virtual_imu/inference/` | Prediction generation and deployable checkpoint inference |
| `src/virtual_imu/viz/` | Shared publication plotting style and figure families |
| `scripts/` | Thin command-line entry points for each pipeline stage |
| `configs/` | Frozen task, split, model, matrix, figure, and report configuration |
| `tests/` | Unit, contract, provenance, and optional local-data integration checks |

## Result lifecycle

A result is considered reproducible only when the following chain is intact:

`raw-source manifest -> prepared-data manifest -> split manifest -> config snapshot -> fit directory -> selected checkpoint -> evaluation predictions/metrics -> aggregate tables -> figures/report -> external package checksums`.

The repository stores source code and protocol files. Raw data, prepared parquet files, checkpoints, run folders, prediction matrices, figures, and complete reports are external artifacts and are intentionally ignored by Git. See [ARTIFACTS.md](ARTIFACTS.md).
