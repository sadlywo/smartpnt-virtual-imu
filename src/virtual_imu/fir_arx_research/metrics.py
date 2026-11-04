"""Physical signal metrics used consistently across all FIR/ARX experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

GYRO = slice(0, 3)
ACCEL = slice(3, 6)


@dataclass(frozen=True)
class SignalMetrics:
    gyro_vector_rmse: float
    accel_vector_rmse: float
    gyro_vector_mae: float
    accel_vector_mae: float
    gyro_bias_norm: float
    accel_bias_norm: float
    gyro_lowpass_rmse: float
    accel_lowpass_rmse: float
    gyro_highpass_rmse: float
    accel_highpass_rmse: float
    combined_rmse: float


def _vector_rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(error**2, axis=1))))


def _vector_mae(error: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(error, axis=1)))


def _band_components(
    values: np.ndarray,
    groups: list[np.ndarray],
    *,
    rate_hz: float,
    cutoff_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    low = np.empty_like(values, dtype=np.float64)
    sos = butter(4, cutoff_hz, btype="lowpass", fs=rate_hz, output="sos")
    for positions in groups:
        low[positions] = sosfiltfilt(sos, values[positions], axis=0)
    return low, values - low


def signal_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    frame: pd.DataFrame | None = None,
    rate_hz: float = 40.0,
    band_cutoff_hz: float = 0.5,
) -> SignalMetrics:
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.shape != ref.shape or pred.ndim != 2 or pred.shape[1] != 6:
        raise ValueError("Prediction and reference must be equal-shaped N x 6 arrays")
    if not np.isfinite(pred).all() or not np.isfinite(ref).all():
        raise ValueError("Prediction/reference must be finite")
    error = pred - ref
    if frame is None:
        groups = [np.arange(len(pred))]
    else:
        groups = list(frame.groupby("sequence_id", sort=False).indices.values())
    low_pred, high_pred = _band_components(pred, groups, rate_hz=rate_hz, cutoff_hz=band_cutoff_hz)
    low_ref, high_ref = _band_components(ref, groups, rate_hz=rate_hz, cutoff_hz=band_cutoff_hz)
    low_error = low_pred - low_ref
    high_error = high_pred - high_ref
    return SignalMetrics(
        gyro_vector_rmse=_vector_rmse(error[:, GYRO]),
        accel_vector_rmse=_vector_rmse(error[:, ACCEL]),
        gyro_vector_mae=_vector_mae(error[:, GYRO]),
        accel_vector_mae=_vector_mae(error[:, ACCEL]),
        gyro_bias_norm=float(np.linalg.norm(error[:, GYRO].mean(axis=0))),
        accel_bias_norm=float(np.linalg.norm(error[:, ACCEL].mean(axis=0))),
        gyro_lowpass_rmse=_vector_rmse(low_error[:, GYRO]),
        accel_lowpass_rmse=_vector_rmse(low_error[:, ACCEL]),
        gyro_highpass_rmse=_vector_rmse(high_error[:, GYRO]),
        accel_highpass_rmse=_vector_rmse(high_error[:, ACCEL]),
        combined_rmse=float(
            np.sqrt(
                np.mean(
                    np.sum(error[:, GYRO] ** 2, axis=1) / max(_vector_rmse(ref[:, GYRO]) ** 2, 1e-12)
                    + np.sum(error[:, ACCEL] ** 2, axis=1)
                    / max(_vector_rmse(ref[:, ACCEL]) ** 2, 1e-12)
                )
            )
        ),
    )


def identity_normalized_score(
    prediction: np.ndarray,
    reference: np.ndarray,
    identity: np.ndarray,
    *,
    frame: pd.DataFrame | None = None,
) -> float:
    candidate = signal_metrics(prediction, reference, frame=frame)
    baseline = signal_metrics(identity, reference, frame=frame)
    ratios = [
        candidate.gyro_vector_rmse / max(baseline.gyro_vector_rmse, 1e-12),
        candidate.accel_vector_rmse / max(baseline.accel_vector_rmse, 1e-12),
    ]
    return float(np.mean(ratios))


def improvements(metrics: SignalMetrics, baseline: SignalMetrics) -> dict[str, float]:
    result = {}
    for name in (
        "gyro_vector_rmse",
        "accel_vector_rmse",
        "gyro_vector_mae",
        "accel_vector_mae",
        "gyro_highpass_rmse",
        "accel_highpass_rmse",
    ):
        value = getattr(metrics, name)
        base = getattr(baseline, name)
        result[f"{name}_improvement_pct"] = float(100.0 * (base - value) / max(base, 1e-12))
    return result
