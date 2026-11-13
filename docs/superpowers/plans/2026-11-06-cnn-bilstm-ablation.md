# Shared CNN-BiLSTM Ablation Implementation Plan

> Execute in this chat using bounded workers with disjoint files; user has authorized implementation, local CUDA training and final reporting.

**Goal:** Finish138 seeded fits isolating temporal CNN, recurrence/direction, context and two capacity controls under shared ALiMU and joint ALiMU/Penn training.

**Architecture:** New `virtual_imu.ablations` package imports the frozen-equivalent CNN-BiLSTM class without editing prior sealed code. Common320-point anchors produce identical center40 labels for all window lengths. Training workers consume frozen YAML, save artifacts and lock all checkpoints before the coordinator loads TEST. Reporting/verification are separate from fitting.

**Stack:** pinn_imu Python3.9, PyTorch2.3 CUDA12.1, pandas/parquet, YAML, NumPy, existing IMU figure style.

- [x] Audit prior CNN-BiLSTM evidence and pin selected historical configs/source.
- [x] Add meaningful failing model/data tests; implement treatment construction and common supports; pass those tests.
- [x] Implement frozen YAML, deterministic balanced training, early stopping, checkpoint lock, common-support prediction and provenance.
- [x] Run CUDA finite-gradient/real-data preflight, estimate throughput, fix observed failures before formal run.
- [x] Execute all138 fits with fixed80-epoch maximum and checkpoint selection on VALID only; preserve resumable completed jobs and progress.
- [x] Freeze fit registry; then evaluate all TEST outputs and export complete physical/per-group results, paired deltas, interactions and compute costs.
- [x] Generate paper-style figures, Chinese findings and technical report; independently replay exported predictions and numerical summaries.
- [x] Inspect figures, verify all hashes/config/source/data links, report measured results and limitations. No commit or publishing requested.

Completion evidence: `runs/cnn_bilstm_shared_ablation_20261106/verification.json` passes138 checkpoints/300 predictions/15,360 summary rows; `independent_review.json` passes table and interpretation review with no unresolved P1/P2; `preparation/final_regression.json` records105 passed tests; `figures/visual_review.json` records6 reviewed bundles/18 exports. Final artifact identity is sealed by `artifact_manifest.json` and `complete.json`. See `interpretation.md` for the VALID/TEST and main/supplemental distinctions.

File ownership: model worker owns `src/virtual_imu/ablations/models.py` and `tests/test_cnn_bilstm_ablation_models.py`; coordinator owns package init, `data.py`, `training.py`, `pipeline.py`, runner/config and data/pipeline tests. Reporting worker can later own `reporting.py` and `viz/plots_cnn_ablation.py` once table schemas are stable. Keep old frozen source files unmodified.
