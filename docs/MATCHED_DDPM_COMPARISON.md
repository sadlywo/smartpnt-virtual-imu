# Matched ALiMU model comparison

GitHub阅读：[最新结果汇总](results/2026-11-13/README.md) · [完整报告、CSV与图表](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113)。本页工件链接指向Release，下载包内按原run名称查找；本地源路径仍用代码形式标识。


This experiment reruns five families independently for BMX160, MPU9250 and
LSM9DS1, each mapped to the measured VN100 reference. It preserves the existing
recording-disjoint `unified_imu_v2` split and all prior run directories.

```powershell
$env:PYTHONPATH='src'
$env:PYTHONUTF8='1'
& 'D:/Anaconda3/envs/pinn_imu/python.exe' scripts/run_matched_comparison.py --config configs/ddpm/alimu_matched.yaml --run-id alimu_matched_ddpm_20261106_final
```

Add `--resume` only when continuing the same frozen run. Source, config,
manifest and completed checkpoint/prediction hashes must still match; interrupted
attempts are archived and restarted from their fixed seeds.

Common settings: 80 samples at 40 Hz, stride40 with tail coverage, seven input
channels (six LOW signals and a constant missing-temperature channel), train-only
feature/residual normalization, Adam, batch64, maximum40 epochs, patience7,
minimum validation improvement .001, AMP and gradient clipping. Three declared
candidate configurations are selected on seed20261106 using the same32 stitched
validation windows; final seeds are20261106/20261107/20261108. All method/pair
choices are locked before any TEST read. The first seed reuses its selected
search checkpoint. Two isolated training workers use the same local CUDA device.
AMP overflow uses GradScaler's skipped-step/scale-backoff recovery; each epoch
records `amp_skipped_steps`. FP32 training still rejects nonfinite gradients.

Methods: CNN-BiLSTM and Informer reuse the historical common-task adapters;
ordinary Transformer predicts standardized HIGH-minus-LOW directly; DDPM uses
100 cosine forward/reverse steps, x0 prediction and four Monte Carlo draws.
The no-diffusion adapter retains every DDPM denoiser parameter, fixes the noisy
input and diffusion time to zero, and learns the same residual with direct MSE.
Both its independently tuned configuration and a strict DDPM-configuration
ablation are evaluated. Strict pairs have identical initial backbone hashes.

Every stochastic method reports all three seeds; raw/affine controls are fitted
once. Summary tables keep gyro and acceleration units separate and include both
component RMSE and 3D vector RMSE (`sqrt(3)` times component RMSE). Dataset-level
metrics pool sample errors before taking the square root. Per-seed differences
between DDPM and its strict ablation retain both improvements and regressions.
CUDA latency includes the actual sampling chain and draws; throughput amortized
over batch64 is reported separately from batch1 request latency.

The authoritative directory is `runs/alimu_matched_ddpm_20261106_final/`:
`config.yaml`, `provenance.json`, `source_snapshot/`, pair manifests/statistics,
`fits/`, `selections.lock.json`, `evaluations/`, `metrics_per_seed.csv`,
`summary.csv`, `matched_seed_deltas.csv`, `latency.csv`, `report.md`, and paper
figures in PNG/SVG/PDF. `complete.json` is written only after the whole matrix,
report and checksums exist. `outputs/latest/` is not used.

This is a retrospective, fixed-data comparison. Three optimization seeds do not
create independent recordings; the TEST split contains two physical recordings
shared between the three LOW devices. VN100 is a measurement reference. The
experiment does not establish global SOTA, absolute sensor grade, causal online
operation, or navigation performance.

The completed local run and independently verified interpretation are available
in [findings.md](https://github.com/sadlywo/smartpnt-virtual-imu/releases/tag/research-results-20261113).
All 81 fits and 54 final slots completed; 41 checks passed. Informer had the
lowest pooled gyro RMSE, and independently tuned no-diffusion regression had
the lowest acceleration RMSE. All 18 strict paired deltas favored removing
diffusion. Original and repeated latency tables are retained in that run.
