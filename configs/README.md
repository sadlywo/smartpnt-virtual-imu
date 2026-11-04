# Configuration Index

All repeatable experiments are defined by YAML. Configs are versioned; generated snapshots are stored with their corresponding run directory.

| Config family | Examples | Purpose |
|---|---|---|
| Simulation and smoke | `sim_baseline.yaml`, `train_tcn.yaml` | Self-contained data generation, training, and pipeline validation |
| SmartPNT data | `smartpnt_example.yaml`, `smartpnt_gt_supervised.yaml` | Raw-file inspection, IMU/GT alignment, and navigation labels |
| Multi-dataset acquisition | `multi_dataset_pilot_download.yaml`, `multi_dataset_strict_prepare.yaml` | Public pilot download and calibrated preparation |
| Unified dataset | `unified_imu_dataset.yaml`, `check_unified_training.yaml` | 40 Hz paired/navigation tensors, archive deduplication, and split validation |
| ALiMU | `alimu_*` | Leave-sensor models, context ablations, baselines, and three-seed matrix |
| PennCOSYVIO | `penn_*` | Leave-one-sequence-out folds, calibrations, comparisons, and seeds |
| Formal pipelines | `train_formal_*`, `train_benchmark_matrix.yaml` | Frozen paired and navigation training matrices |
| Reviewer benchmark | `reviewer_hpo.yaml`, `reviewer_splits.yaml`, `reviewer_shared_classical.yaml` | Hyperparameter selection, shared controls, latency, and ranking audits |
| Figures and reports | `reviewer_figures.yaml`, `reviewer_report.yaml`, `figures/*.yaml` | Publication exports, source-data indexes, captions, and report assembly |
| Server execution | `server_environment.yaml`, `reviewer_export*.yaml` | GPU preflight, packaging, queueing, and result export |

Rules:

- Do not select hyperparameters with test metrics.
- Keep data paths relative to the project root where practical.
- Change a frozen protocol by creating a new config file and recording a new experiment ID.
- Store the resolved config snapshot under the resulting `runs/` directory.
- Figure and report configs must reference an immutable result package rather than `outputs/latest/` alone.
