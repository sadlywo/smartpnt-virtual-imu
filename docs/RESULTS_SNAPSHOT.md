# Results Snapshot

Latest numerical update: [2026-11-13 ALiMU results](results/2026-11-13/README.md), including matched FIR/ARX controls, context and future-information experiments, device-held-out results, source hashes, and the GitHub Release download. The sections below retain the earlier benchmark framework snapshot and its original scope.

This page describes the scope of the completed benchmark represented by the frozen source and protocol. Raw metrics, checkpoints, predictions, figures, and full reports are external artifacts and are not stored in Git.

## Frozen benchmark inventory

- 93 selected protocol/model training cells;
- 558 formal task/seed evaluations;
- 3 optimization seeds per neural selection cell;
- 120 cross-domain signal-transfer model records;
- 5 Penn optical-ground-truth downstream views over 4 physical recordings;
- shared-input classical controls, known-device bank controls, calibration baselines, parameter audits, and CPU/GPU inference measurements;
- publication-style figures and a fully indexed report generated from saved predictions and tables.

The reported figure inventory includes signal time series, spectra and coherence, segment integrals, training histories, metric comparisons, rankings, downstream diagnostics, transfer results, parameter summaries, and latency summaries. Published figures are generated through `src/virtual_imu/viz` and `.agents/skills/imu-research-visualization`.

## Interpretation boundaries

- Neural backbones such as DeepGyro, TLIO, TartanIMU, Informer, PatchTST, and iTransformer are common-task adaptations, not complete replications of each original system.
- SmartPNT-MSF supplies no HIGH six-axis label for navigation. Its result is a short-window endpoint displacement task, not full trajectory ATE/RPE.
- The paired task uses no temperature input and no independent static branch.
- FusionPortable was reviewed for alignment and quality but is not included in the frozen formal paired benchmark.
- The classical known-device bank and the neural shared-input mapping have different information scopes.
- Seed sample SD measures optimizer variability on a fixed split and is not a recording-level confidence interval.
- Causal, window, and noncausal methods are separated; composite scores that add gyro and accelerometer error are not used.
- Navigation and signal metrics use different physical units and are never pooled into one score.

## Where to inspect a result

For a locally produced run, use its `runs/<experiment_id>/` directory and verify:

- the config snapshot and Git commit;
- the prepared-data manifest hash;
- train/validation/test membership and deduplication records;
- selected checkpoint and its hash;
- evaluation predictions and generated tables;
- figure-source CSVs and `figure_index.json`;
- the report and its QA receipt.

The repository does not claim that a public user can access the maintainer's external artifact store automatically. Reproduce the data preparation and training path, or request the package identified in the release notes, before comparing numerical results.
