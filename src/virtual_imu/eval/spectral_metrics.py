"""Frequency-domain metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import coherence, csd, welch

from virtual_imu.eval.signal_metrics import ACCEL_AXES, GYRO_AXES


def spectral_metrics(df: pd.DataFrame, pred_prefix: str, ref_prefix: str = "high") -> dict[str, float]:
    """Compute compact PSD/coherence/phase metrics."""
    metrics: dict[str, float] = {}
    dt = float(np.nanmedian(np.diff(df["timestamp_s"]))) if len(df) > 1 else 0.01
    fs = 1.0 / max(dt, 1e-9)
    nperseg = min(256, max(16, len(df) // 4))
    for sensor, axes in {"gyro": GYRO_AXES, "accel": ACCEL_AXES}.items():
        band_errors = []
        hf_ratios = []
        coherences = []
        phase_delays = []
        for axis in axes:
            pred = df[f"{axis}_{pred_prefix}"].to_numpy(dtype=float)
            ref = df[f"{axis}_{ref_prefix}"].to_numpy(dtype=float)
            freq, psd_pred = welch(pred, fs=fs, nperseg=nperseg)
            _, psd_ref = welch(ref, fs=fs, nperseg=nperseg)
            band_errors.append(float(np.sqrt(np.mean(np.square(np.log1p(psd_pred) - np.log1p(psd_ref))))))
            hf = freq > 0.25 * fs
            if np.any(hf):
                hf_ratios.append(float(np.mean(psd_pred[hf]) / (np.mean(psd_ref[hf]) + 1e-12)))
            if np.nanstd(pred) > 1e-12 and np.nanstd(ref) > 1e-12:
                with np.errstate(invalid="ignore", divide="ignore"):
                    _, coh = coherence(pred, ref, fs=fs, nperseg=nperseg)
                finite_coh = coh[np.isfinite(coh)]
                if finite_coh.size:
                    coherences.append(float(np.mean(finite_coh)))
            f_csd, pxy = csd(pred, ref, fs=fs, nperseg=nperseg)
            valid = f_csd > 1e-6
            if np.any(valid):
                phase = np.unwrap(np.angle(pxy[valid]))
                phase_delays.append(float(np.nanmedian(-phase / (2.0 * np.pi * f_csd[valid]))))
        metrics[f"{sensor}_frequency_band_rmse"] = float(np.nanmean(band_errors))
        metrics[f"{sensor}_high_frequency_noise_ratio"] = float(np.nanmean(hf_ratios)) if hf_ratios else float("nan")
        metrics[f"{sensor}_coherence"] = float(np.nanmean(coherences)) if coherences else float("nan")
        metrics[f"{sensor}_phase_delay_s"] = float(np.nanmean(phase_delays)) if phase_delays else float("nan")
    return metrics
