# Reviewer HPO and Generalization Implementation Plan

> **For agentic workers:** Use independent subagents for attention backbones, protocol splits/HPO orchestration, physical downstream validation, and performance measurement. Preserve existing immutable results and all user changes. Do not commit datasets, runs, checkpoints or generated figures.

**Goal:** Address all five reviewer concerns with actual server experiments: validation-only tuning, dataset-specific training and ranking, scene/domain transfer, measured complexity/latency, and independent-GT downstream evaluation.

**Architecture:** A new run family freezes the search space, derived physical-recording splits and all source/data hashes. Search does not evaluate test data. Selected settings are retrained with three independent optimization seeds, then evaluated once. Cross-domain evaluations reuse source-only selected models without fitting anything to target data. Physical downstream propagation uses independent optical poses only for initial state and scoring.

**Tech Stack:** Existing Python 3.12 / PyTorch 2.4.0+cu121 / NumPy 1.26.4, RTX 4090, YAML configurations, immutable unified_imu_v2 data, Python/scipy SO(3), project/Nature scientific plotting.

## Requirements and proof of completion

| User requirement | Concrete experiment | Completion evidence |
|---|---|---|
| Tune large time-series methods | Full Transformer, Informer, PatchTST and iTransformer, alongside all previous neural controls; 24 candidates each, 8/24/80 epoch successive halving, two validation seeds for final choice, three fresh final seeds | Frozen candidates, every rung/failed trial/budget, validation-only selection record, final finite checkpoints and predictions |
| Dataset effects on ranking | Independently tune/train pooled, ALiMU-only and Penn-only; retain separate physical gyro/accel metrics and device mappings | Per-source/per-test dataset tables, rank correlations, seed and recording variation; distinguish this from slicing a pooled model |
| Cross-scene capability | Nested selection in four Penn recording folds and three SmartPNT trajectory rotations; AL dyn08/dyn09 separately; AL↔Penn zero-shot with six input channels | Split disjointness by physical recording, no target statistics/calibration, source-only selection and transfer predictions |
| Parameters/inference speed | Count actual selected checkpoint parameters/buffers; batch1 latency and batch throughput on the same RTX4090 and CPU, FP32, warmup, synchronized timing | Raw repeated timing CSV, p50/p95, mean/SD, device/thread/precision settings, parameter and memory measurements |
| Useful downstream application | Penn optical AprilTag GT: SO(3) attitude errors and six-axis strapdown errors over fixed 1.975/4.975/9.975/19.975 s supported windows | Hash-pinned independent GT, support/clock/rig QC, no fitted test alignment, all methods and windows retained, genuine geodesic/position error calculations |

Improvement is an empirical outcome, not a completion condition that may be manufactured. Preserve failures and absence of downstream gains. These are common-task encoder/backbone adaptations, not full reproductions of every original forecasting or navigation system. Prior test results are already known, so comparisons remain retrospective.

## Task 1: Freeze data and trial controls

- [x] Generate derived metadata with `scripts/prepare_reviewer_splits.py --config configs/reviewer_splits.yaml`; never rewrite original data or manifest. Local and server protocol indexes include the nine primary protocols and separate AL↔Penn transfer assignments.
- [x] Verify physical-recording block assignments, including all three simultaneous AL sensors and exact Traj1/Traj2 deduplication.
- [x] Materialize `configs/reviewer_hpo.yaml` to a plan-only job contract. Verified: 93 model/protocol cells, 2,232 initial trial configurations, 3,348 search/confirmation fits, plus 279 tuned and 279 matched-default final refits (558 final evaluations; 3,906 total planned fits).
- [x] Search uses six LOW channels consistently for source-only and pooled models. This deliberately separates new matched controls from historical nine-channel models.
- [x] Main paired windows are validation-selected among 80/160/320 points. Actual paired segments exceed 320 samples and complete stride-40 windows share covered physical rows; navigation remains 80 points and 1.975-second labels. The evaluator enforces these conditions.

## Task 2: Attention and training support

- [x] Integrate independently implemented Informer ProbSparse/distillation, PatchTST shared patch encoding and iTransformer variate encoding; test actual sparse selection, finite gradients, repeated inference and checkpoint restoration.
- [x] Expose Transformer FF width, activation and pre/post norm while preserving original constructor defaults and checkpoint compatibility.
- [x] Add opt-in train/validation-only loading and disable built-in navigation test pass during every search and training phase. Checksum-only provenance validation does not decode held-out payloads.
- [x] Add opt-in warmup/cosine schedule with recorded warmup and minimum LR; keep old experiments unchanged under defaults.
- [x] Complete 50 actual RTX4090 forward/backward/AdamW determinism cases before training, covering default and sampled stress configurations plus 80/160/320 attention contexts. The tested bundle remains frozen. Evidence: `runs/server_reviewer_deployment_20261024/receipts/server_bundle_reviewer_hpo_20261024_v1/runs/server_reviewer_deployment_20261024/model_gpu_preflight.json`.

## Task 3: Selection and final training

- [ ] Deterministic seeded search: 24 candidates, 24→8→2 promotions, budgets 8/24/80 epochs. Promoted runs explicitly restart from scratch under new rung identities.
- [ ] Confirm the top two candidates using one additional seed. Select by mean validation score across the search and confirmation seeds; never by test score.
- [ ] Train each winner and its matched fixed default with seeds 20261010/20261011/20261012; maximum 80 epochs, early stopping 15, AdamW and searched LR/WD/loss/dropout/capacity/batch. Full-budget default refits are independent of whether the default trial survived halving, and test evaluation starts only after all selections are frozen.
- [ ] Tune classical regularization/history lengths on each fold's train/validation data as well; preserve historical FIR/ARX as separate controls.
- [ ] Record failed/OOM candidates. Never silently reuse another candidate or present a partial matrix as complete.

## Task 4: Evaluation and downstream validity

- [x] Generalized evaluator verifies new split/config/model provenance and exact physical sample sets; keep gyro rad/s and accel m/s² separate. New full-matrix measurements remain pending.
- [ ] Produce dataset-dependent ranking tables and descriptive Spearman/Kendall statistics. Small recording counts limit inferential claims; seed SD is not a recording confidence interval.
- [ ] Evaluate AL↔Penn transfer using each source's existing normalization, without target calibration or target validation. Label this as a combined device/reference/scene shift.
- [x] Acquire Penn GT from the pinned official repository and verify actual schemas and clock support. Execute the frozen `configs/downstream_penn.yaml` on all historical Penn-bs models; new tuned/fold results are still pending. Historical output: `runs/reviewer_downstream_historical_20261024/`.
- [x] Use midpoint SO(3) gyro integration and trapezoid world acceleration. Initial pose/velocity from optical GT is an explicit diagnostic oracle; no subsequent GT enters propagation. No cumulative overlapping displacement predictions. Independent frame/clock review: `runs/hpo_research_20261024/downstream_independent_review.json`.

## Task 5: Server execution and final deliverables

- [ ] Measure existing and selected models using `scripts/benchmark_forward_latency.py`; separate device forward timing, CPU wall latency and end-to-end copy timing. Historical 15 distinct task/model structures have completed 30 GPU batch records and 15 single-thread CPU records; actual selected/default structures remain pending. Fixed seed 20261010 avoids redundant timing of identical structures and is independent of accuracy rankings.
- [x] Export a fresh immutable source/data bundle and separately hash the small optical-GT supplement. Reconnect using the previously authorized endpoint and interactive credential only. Bundle SHA256 `870745185b756dbcb34337be9fa6432af7c50959f035b1dad822e45ea1fab981`; optical supplement SHA256 `66df67028a9256e6317a57462044a4ef346bf251f29e8ccf92e0d299d74de064`.
- [x] Execute under a new tmux session and source-of-truth run; monitor actual process/metrics progress and retain all old results. Exact server run `/root/imu-work/server_bundle_reviewer_hpo_20261024_v1/runs/reviewer_hpo_20261024_122113_893148`, tmux `imu_reviewer`. Actual training started after latency isolation and GPU preflight; this checkbox records launch only, not experiment completion. Three-hour monitor id `imu`.
- [ ] Download/check all selected models, predictions, search histories, timing records, baselines and logs; verify SHA256.
- [ ] Deliver full Chinese Markdown/PDF report, complete CSV/Markdown comparison, Nature-style figures and a requirement-by-requirement evidence audit. Keep the active goal until all five requirements are actually proven.
