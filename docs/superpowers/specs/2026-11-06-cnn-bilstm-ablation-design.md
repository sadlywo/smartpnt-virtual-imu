# Shared CNN-BiLSTM ablation design

User authorization: implement and execute complete CNN-BiLSTM window/module ablations and report measured results. No additional design approval is necessary within this requested scope. Existing data, commits, frozen sources and experiment artifacts stay intact.

## Questions and frozen reference

Does the shared architecture benefit from temporal convolution, bidirectional recurrence, their combination, and longer context? The Oct31 HPO did not independently isolate these factors. The selected `alimu_only` and `pooled` architecture/configuration are identical: six LOW inputs; residual HIGH-minus-LOW target standardized using TRAIN; hidden CNN64, three kernel5 Conv-BatchNorm-GELU blocks, BiLSTM hidden32 per direction, one recurrent layer, LayerNorm64→Linear64→GELU→Linear6 head, dropout0. AdamW lr0.0022830268403520168, weight_decay0.007082535831697058, batch32, warmup5/cosine to1e-6, max80 epochs, patience15, gradient clip1. CUDA in pinn_imu. Source config hashes and architecture source are pinned before training.

## Factorial treatments

Two protocols: ALiMU three-device shared model; ALiMU+Penn dataset-balanced shared model. Report joint training separately on ALiMU and Penn. No device identity or temperature features. Frozen unified_imu_v2 measured paired records; physical recording groups disjoint; minimum sequence320, trim VALID/TEST to stride40. TRAIN-only statistics.

Primary full factorial: windows40/80/160/320 × variants `cnn_bilstm`, `bilstm_only`, `cnn_only`, `mlp_only`, `cnn_lstm` × three seeds20261010/11/12 × two protocols =120 fits. `bilstm_only` uses a pointwise six-to64 projection, removing temporal CNN; `cnn_only` removes recurrence, retaining the identical64-channel readout; `mlp_only` removes both temporal modules. `cnn_lstm` removes the backward direction (hidden32). All retained blocks with matching shapes inherit the full model's seeded initial weights; document changed parameter counts.

Supplemental320-point controls: `cnn_lstm_wide` (unidirectional hidden64, same output width, more recurrent parameters), `pointwise_bilstm` (three kernel1 blocks, preserving depth/normalization/nonlinearity), `cnn_bilstm_dense` (original dense-window loss). Three controls×three seeds×two protocols=18 fits; total138. Dense-window control is not part of the strictly matched-label factorial.

## Common supports and isolation

Each sequence uses the SAME320-point anchor windows at stride40, excluding incomplete anchors; each anchor's central40 points (offset140:180) is the supervised/evaluated target for every primary and capacity-control window. Smaller windows crop symmetrically around that center. Loss uses only this common target block, yielding identical training labels, sample count, dataset sampling weights, minibatch indices and evaluation timestamps across windows. Prediction blocks do not overlap; exclude sequence edges equally across all treatments. Context40/80/160/320 has equal left/right expansion; models may access future data and are not an online causal claim.

The dense reference alone supervises all320 points, then reports central-block accuracy on the same primary support plus an optional full-window stitched diagnostic. It tests sensitivity to the loss-support convention, not a single module. Absolute results need not reproduce historical whole-window/stitch numbers; historical values never substitute for new controlled baselines.

Seed-blocked dataset-balanced replacement sampling: equal expected dataset mass, pairs not routed. Same local sampling RNG per protocol/seed across variants and lengths. Shuffle execution order with fixed experiment seed to spread machine-load effects. One GPU, at most two workers after smoke measurements. No HPO using TEST. Validation score is mean gyro/accel RMSE-to-Raw ratios per pair, equal pair mean within each dataset and equal dataset mean; checkpoint selected using all common VALID blocks, early stopping same for every treatment. Test signals are loaded only after all fit/checkpoint identities are locked.

## Evidence and outputs

Primary physical component RMSE and vector RMSE, MAE, bias, normalized selection score, per-axis/per-device/per-recording breakdown, seed mean/sampleSD, exact paired-seed relative deltas vs full at each length, window deltas vs320, CNN×recurrent interaction. Use3seeds as optimization repeats, never independent physical recordings or confidence intervals. ALiMU TEST has2 recordings shared by3 devices; Penn bs only1, limiting generalization. Report non-improvements candidly.

Record parameter counts, actual epochs, best epoch, skipped AMP steps, wall time and load-sensitive CUDA timing (synchronization, fixed warmup/repeats). Plots use virtual_imu.viz styling, PNG/SVG/PDF with captions. Run source of truth contains config, source snapshot, git+dirty state, manifest/file hashes, support keys, initialization digests, sampling digests, histories, best checkpoints, predictions, metrics, report, figures, independent verification receipt. Never modify outputs/latest or commit generated data/checkpoints/figures.

Pre-TEST analysis amendment: additionally compare single center-block output against arithmetic overlap averaging for the same selected full/dense320 models (12 fits, no added training). Keep identical primary timestamps and checkpoints; retain full normalized window predictions so the averaging can be independently replayed. This distinguishes inference aggregation from context/training-label effects. It does not reselect checkpoints or recreate historical whole-window validation selection. Analysis-plan v1 and v2 are retained; both predate new TEST access.

Pre-TEST clarification in analysis-plan v4 supersedes the causal interpretation in the preceding paragraph: multi-anchor fusion changes the union of inputs available to a timestamp (up to600 points for eight320-point anchors at stride40) and includes noncentral output positions. It is a fixed-checkpoint, fixed-target diagnostic combining these effects, and cannot isolate pure averaging from context gains. The earlier source/plan versions are retained. Primary training, checkpoint/window selection and metric computations are unchanged.

## Checks

Meaningful tests cover identical central labels/row keys across windows, no sequence/split leakage, dataset-balanced deterministic sampling, exact baseline architecture/output parity, genuinely absent modules, parameter-control direction, preserved shared initialization, training-only normalization and metric pooling. CUDA preflight checks finite gradients and validates real data throughput. Before completion independently replay prediction metrics/seedSD/paired deltas, verify support and checkpoint/source hashes, validate selected epoch minima, inspect all figure layouts.
