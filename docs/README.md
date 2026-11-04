# Documentation Index

Start here when adapting the project, reproducing the benchmark, or auditing a result.

| Document | Purpose |
|---|---|
| [PROJECT_FRAMEWORK.md](PROJECT_FRAMEWORK.md) | Research questions, task definitions, architecture, invariants, and repository map |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Environment setup and the staged workflow from smoke test to full benchmark |
| [ARTIFACTS.md](ARTIFACTS.md) | Data, run, checkpoint, figure, report, and transfer-artifact policy |
| [RESULTS_SNAPSHOT.md](RESULTS_SNAPSHOT.md) | Scope and status of the completed benchmark, with limitations |
| [CI.md](CI.md) | Local validation and the ready-to-enable GitHub Actions template |
| [DATA.md](DATA.md) | Raw data layouts and preparation entry points |
| [TRAINING_PROTOCOL_V2.md](TRAINING_PROTOCOL_V2.md) | Frozen train/validation/test and model-selection protocol |
| [BENCHMARK_EXPANSION_20261024.md](BENCHMARK_EXPANSION_20261024.md) | Expanded neural, classical, navigation, and transfer comparisons |
| [UNIFIED_IMU_DATASET.md](UNIFIED_IMU_DATASET.md) | Unified tensor schema, labels, and archive deduplication |
| [ALIMU_MATRIX_RUN_GUIDE.md](ALIMU_MATRIX_RUN_GUIDE.md) | ALiMU leave-sensor matrix commands |
| [PENN_MATRIX_RUN_GUIDE.md](PENN_MATRIX_RUN_GUIDE.md) | PennCOSYVIO leave-one-sequence-out matrix commands |
| [SERVER_TRAINING.md](SERVER_TRAINING.md) | GPU preflight, packaging, upload, and remote execution |
| [CAD_XSENS_DOT2_WATCH.md](CAD_XSENS_DOT2_WATCH.md) | Reproducible sensor-mount CAD files and build flow |

The repository is intentionally split between versioned source/protocol files and external scientific artifacts. `docs/ARTIFACTS.md` explains that boundary.
