# Matched ALiMU DDPM comparison

The user approved rerunning CNN-BiLSTM, Informer, Transformer and a no-diffusion
Transformer ablation under common conditions. Execute in the existing feature
branch; preserve prior immutable runs and unrelated local files.

## Scientific contract

- Five methods: conditional DDPM, CNN-BiLSTM, Informer, ordinary Transformer,
  and the exact DDPM denoiser used as a direct residual regressor. The last
  retains every denoiser parameter, feeds zeros into the six noisy channels,
  fixes diffusion time to zero, and performs one forward pass without noise.
- Fit each ALiMU LOW sensor separately to VN100 using the frozen v2 manifest.
  Physical recording separation, train-only normalization, feature channels,
  80-point windows, 40-point strides and tail coverage are identical.
- Adam, AMP on local CUDA, batch 64, maximum 40 epochs, patience 7, minimum
  improvement .001, gradient clipping 1, weight decay 1e-5. All direct models
  minimize MSE of standardized HIGH-minus-LOW residuals; DDPM predicts x0 with
  the same standardized residual MSE under its forward corruption objective.
- Each method/pair searches three preregistered configurations on seed
  20261106, selected by the same 32 fixed validation windows, stitched before
  scoring mean relative gyro/accel RMSE. No TEST values enter selection.
- Final evaluation uses seeds 20261106, 20261107, 20261108. The first seed's
  selected search checkpoint is reused rather than retrained identically.
  Selected hyperparameters stay fixed for the other two seeds.
- Strict ablation additionally trains the no-diffusion model using the exact
  DDPM-selected architecture, learning rate and initialization seed for each
  pair/seed. If its own search selects a different configuration, both the
  independently tuned and matched-DDPM-configuration results are retained.
- All five method/pair selections are written before any final TEST read.
  Validation/test use the same row support and overlap averaging. DDPM uses
  100 ancestral steps and 4 independent draws; deterministic direct models
  use one pass (repeating a deterministic pass adds no information).
- Compare physical gyro/accel component and 3D vector RMSE separately,
  per-record and per-pair, then pooled over common timestamps. Report all
  three seeds, their mean/sample SD, and matched-seed ablation differences.
  Optimization seeds do not represent independent recordings.
- Measure CUDA latency after warmup using batch=1 and batch=64, including all
  DDPM draws/steps; disclose window context separately from computation time.

## Artifacts and verification

YAML config, exact source snapshot/hashes, manifest and normalization, trial
histories, checkpoint SHA256, locked selections, per-seed predictions/metrics,
matched-seed deltas, latency, tables, figures and report all live under one new
`runs/` directory. Resume must reject changes in source/config/manifest and
archive incomplete trial attempts. Required tests cover exact ablation weight
compatibility, no-noise direct inference, shape/backprop for all architectures,
selection-before-TEST ordering, fixed validation support and resume rejection.
Plots use `src/virtual_imu/viz` paper styling. No generated data are committed.
