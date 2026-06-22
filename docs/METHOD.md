# Method

The core task is sequence-to-sequence regression from low-grade IMU observations to high-grade IMU targets.

Input features:

- Gyroscope: `gx_low`, `gy_low`, `gz_low`
- Accelerometer: `ax_low`, `ay_low`, `az_low`
- Optional temperature: `temperature_c`

Targets:

- High-grade gyroscope: `gx_high`, `gy_high`, `gz_high`
- High-grade accelerometer: `ax_high`, `ay_high`, `az_high`

Baselines include identity, affine calibration, and classical filtering. Learning models include TCN, GRU, Transformer encoder, and 1D U-Net. The default model is TCN because it is fast, causal-friendly, and robust for short IMU windows.

Loss terms combine signal loss with optional derivative, spectral, Allan-inspired multi-scale, and preintegration consistency penalties.
