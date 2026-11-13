# CNN-LSTM Module Replacement Implementation Plan

> Execute the user's authorized experiment in this chat with disjoint model/report workers and local orchestration. No commit or publication requested.

**Goal:** Evaluate CNN front-end and temporal back-end replacements with48 new seeded fits plus18 verifiable reused controls; report scope-specific strengths and tradeoffs.

**Architecture:** New `virtual_imu.replacements` package reuses frozen common-support data and numerical helpers without changing sealed files. Reference-compatible replacement factory, explicit borrowed checkpoint records, fixed TEST gate, separate reporting and replay verification.

**Tech Stack:** pinn_imu Python3.9/PyTorch2.3 CUDA12.1, NumPy/pandas/YAML, repository Matplotlib style.

**Current status:** Training66/66 (48 new fits +18 reused controls), VALID/TEST evaluation and all132 files' full-sample independent CUDA numerical replay have passed; seven real figure groups /21 exports passed QA. Independent interpretation review passed; complete.json and artifact_manifest.json close this run. VALID selects CNN-GRU for AL-only and CNN-LSTM64 for the joint model; TEST minima remain descriptive.

Evidence: [user guide](../../CNN_LSTM_MODULE_REPLACEMENT.md), [formal report](../../../runs/cnn_lstm_module_replacement_20261107/report.md), [interpretation](../../../runs/cnn_lstm_module_replacement_20261107/interpretation.md), [real figure index](../../../runs/cnn_lstm_module_replacement_20261107/figures/figure_index.md), [numerical verification](../../../runs/cnn_lstm_module_replacement_20261107/verification.json), [figure QA](../../../runs/cnn_lstm_module_replacement_20261107/figures/visual_review.json).

- [x] Model worker: write failing tests for `build_replacement_model(variant, seed, model_config, replacement_config)` in `tests/test_module_replacement_models.py`; implement only `src/virtual_imu/replacements/models.py`; check exact three legacy controls, retained module states/RNG, all11 finite outputs/backprop, parameter budgets, pointwise andTCN dependencies. Run the new model tests with `PYTHONPATH=src`.
- [x] Coordinator: add package init and frozen YAML; implement explicit import records and training/checkpoint gates in `replacements/training.py`, `replacements/pipeline.py`, runner. Reuse old data/common support, preserve original old artifacts. Meaningful failing regression tests in `tests/test_module_replacement_pipeline.py` cover bad source hash/config, partial fits, TRAIN-only and lock rules.
- [x] Run actual CUDA all-new-model finite-gradient/throughput preflight in an independent `_preflight` directory. Pin source/config/data/control hashes and initialization/cost matrix; fix any observed failures before starting formal fits. Freeze randomized66-job schedule with48 new and18 imported.
- [x] Train48 conditions with the bounded two-worker GPU pool, fixed80-epoch ceiling, three seeds and balanced minibatch draws; verify18 imports and complete all66 identities. Preserve interruption diagnostics and resume only matching completed fits.
- [x] Freeze analysis before TEST, lock checkpoint/initialization/config/job identities, persist TEST access and evaluate fixed320/center40 VALID/TEST outputs;66 conditions and132 prediction files complete.
- [x] Reporting worker: own `scripts/report_module_replacement.py`, `src/virtual_imu/viz/plots_module_replacement.py`, `tests/test_module_replacement_reporting.py`; summarize physical metrics and all paired changes vs `cnn_lstm64`, distinguish front/back family and reused/new costs; generate figures and factual Chinese report. Freeze analysis source before TEST.
- [x] Coordinator: independent verification script replays original supports/predictions, checkpoint minima, sampler order, init receipts, source/config/import identities and aggregate arithmetic; test with synthetic or deliberately corrupted fixtures before reading formal TEST.
- [x] Execute formal independent numerical verification: all132 VALID/TEST files and all samples replayed from the66 checkpoints on CUDA; verification.json records PASS.
- [x] Inspect all7 real figure groups /21 PNG/SVG/PDF exports; actual-data visual QA and hash checks PASS.
- [x] Update the user guide and README with verified real tables, VALID choices, paired percentages/SD, domain tradeoffs, recording counts and capacity/context limits; preserve structural theory and original-paper references.
- [x] Complete independent interpretation review and final closure checks; bind the final document/source/output SHAs (coordinator/Pasteur).
- [x] Coordinator: finish final closure/seal and create complete.json / artifact_manifest.json. Final SHA-bound completion is recorded in complete.json.

Reproduction uses a new run, not the current directory being sealed. First copy
`configs/ablations/cnn_lstm_module_replacement.yaml` to
`configs/ablations/cnn_lstm_module_replacement_rerun.yaml`, set its
`experiment_id: cnn_lstm_module_replacement_rerun`, and choose a fresh directory
if that name is already used. The examples below assume that edit is complete.
Keep the original sealed SOURCE as the configured read-only reuse input.
Training, evaluation, cost measurements, reports and verification all write to
the new run; do not recompute or overwrite the current run's costs or reports.
These are reproduction instructions, not commands executed by this doc update.

```powershell
Write-Output 'READS copied YAML and pinned inputs; WRITES new rerun directory only; SKIPS overwriting current/old sealed artifacts'
$env:PYTHONUTF8 = '1'
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$rerunConfig = 'configs/ablations/cnn_lstm_module_replacement_rerun.yaml'
$rerunDir = 'runs/cnn_lstm_module_replacement_rerun'
$pytestTemp = Join-Path 'runs/pytest_tmp' ('module_replacement_' + [guid]::NewGuid().ToString('N'))
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 -m pytest tests/test_module_replacement_models.py tests/test_module_replacement_pipeline.py tests/test_module_replacement_reporting.py tests/test_module_replacement_verification.py --basetemp $pytestTemp -p no:cacheprovider -q
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/run_module_replacement.py --config $rerunConfig --run-dir $rerunDir --mode preflight
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/run_module_replacement.py --config $rerunConfig --run-dir $rerunDir --mode train
# Freeze/review the new run's analysis before opening its TEST phase.
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/run_module_replacement.py --config $rerunConfig --run-dir $rerunDir --mode evaluate
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/report_module_replacement.py --run-dir $rerunDir
& D:/Anaconda3/envs/pinn_imu/python.exe -X utf8 scripts/verify_module_replacement.py --run-dir $rerunDir
```
