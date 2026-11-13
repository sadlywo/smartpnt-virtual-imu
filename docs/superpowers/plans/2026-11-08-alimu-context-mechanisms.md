# ALiMU Context and Mechanisms Implementation Plan

> **For agentic workers:** Execute sequentially in this task using the executing-plans workflow. User approval already covers this design; no new approval step or subagent is needed.

**Goal:** Run the authorized four-stage ALiMU study and produce traceable results, causal controls, training intermediate figures and exact grouped Shapley explanations.

**Architecture:** Add a separate context_study package, import sealed loaders/model builders without editing them, and run a resumable two-worker CUDA coordinator. Freeze selection before TEST. Keep numerical artifacts under a new runs directory and plotting under src/virtual_imu/viz.

**Tech Stack:** Python pinn_imu, PyTorch CUDA, NumPy/Pandas, YAML, Matplotlib.

## File map and tasks
- [x] Tests: tests/test_context_study.py covers future perturbation in train/eval, strict sensor/recording separation, identical window target support, exact Shapley completeness, and test-free selection.
- [x] Models: src/virtual_imu/context_study/models.py adds causal Conv/normalization, bounded BiGRU, delayed backends and parameter-tracked mechanism variants; existing frozen factories provide A/D/E models.
- [x] Data: src/virtual_imu/context_study/data.py filters fold metadata before loading and fits normalization only on source TRAIN.
- [x] Training: src/virtual_imu/context_study/training.py logs epoch diagnostics, snapshots, full RNG/optimizer resumability and checkpoint hashes.
- [x] Pipeline: src/virtual_imu/context_study/pipeline.py freezes inputs, validates exact reuse, runs staged jobs, locks VALID choices, evaluates and exports reports. scripts/run_context_study.py is the logged CLI.
- [x] Explain: src/virtual_imu/context_study/explain.py enumerates exact Shapley coalitions with train backgrounds and exports numeric completeness receipts and intermediate arrays.
- [x] Figures/report: src/virtual_imu/viz/plots_context_study.py exports SVG/PDF/PNG with captions and source tables. Document questions and findings in the run report and docs/ALIMU_CONTEXT_STUDY.md.
- [x] Verify meaningful tests first fail, implement them, run CUDA preflight, execute all matrices, verify traces/counts/causality/selection/reuse/source hashes, inspect final images, then report findings in Chinese. No commit or push requested.

Runtime update (2026-11-09): original 215 slots plus 12 pretest controls = 227 slots, including 30 exact reuses. The two bounded helper processes were stopped on November 9 at 01:05; a single four-worker coordinator now resumes their full optimizer/RNG states. The frozen YAML retains workers=2; execution_pool_override.json records actual workers=4 and the execution-only cache policy. Training wall times contain contention and are not independent speed benchmarks.

Completion verified (2026-11-09): numerical run sealed at12:09:55 with227slots,197new fits,30exact reuses and213evaluated checkpoints. Independent NumPy auditing recomputed426prediction-role arrays, maximum metric error4.44e-16. All48trained-checkpoint causal receipts passed. Original52study tests and the additional causal-selection/numerical-recovery/report regression checks passed (the report fixture required an explicit local pytest temp directory after Windows denied the default temp root). All111frozen training/execution/analysis/precision source hashes match. The final Informer held-MPU9250 seed required a pre-TEST finite-input FP16 overflow amendment: conditional FP32 sparse-attention recomputation,85events, unchanged architecture/data/seed/budget. Evidence and source hashes are sealed. All2691numerical artifact hashes verified twice after figure revisions. Final presentation run runs/alimu_context_mechanisms_20261109_figures_final contains9reviewed figures/27exports, corrected mechanism-label density, visible Raw LOW generalization baselines, unclipped SHAP labels, a conclusions.md and a120-entry results archive. It adds0training fits and preserves numerical CSV/answers bytes. No remaining requested execution work; no commit/push performed.
