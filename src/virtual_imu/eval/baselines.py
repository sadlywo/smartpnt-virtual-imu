"""Three-tier baseline suite for low-to-high IMU calibration."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from virtual_imu.eval.allan import allan_summary
from virtual_imu.eval.navigation import integrate_dead_reckoning, navigation_metrics
from virtual_imu.eval.preintegration import preintegration_metrics
from virtual_imu.eval.signal_metrics import signal_metrics, snr_improvement
from virtual_imu.eval.spectral_metrics import spectral_metrics
from virtual_imu.io.parquet_store import write_parquet
from virtual_imu.models.baseline_affine import AffineCalibration
from virtual_imu.models.baseline_filters import butterworth_lowpass, moving_average, savitzky_golay

LOW_COLUMNS = ["gx_low", "gy_low", "gz_low", "ax_low", "ay_low", "az_low"]
TARGET_COLUMNS = ["gx_high", "gy_high", "gz_high", "ax_high", "ay_high", "az_high"]
COMPETITOR_COLUMNS = [
    "gx_competitor",
    "gy_competitor",
    "gz_competitor",
    "ax_competitor",
    "ay_competitor",
    "az_competitor",
]
VIRTUAL_COLUMNS = ["gx_virtual", "gy_virtual", "gz_virtual", "ax_virtual", "ay_virtual", "az_virtual"]


@dataclass(frozen=True)
class BaselineMetadata:
    """Structured baseline metadata for fair comparison tables."""

    name: str
    layer: str
    family: str
    causal: bool
    uses_future: bool
    trained: bool
    output: str
    public_reference: str
    note: str


IMPLEMENTED_BASELINES: dict[str, BaselineMetadata] = {
    "identity": BaselineMetadata(
        "identity",
        "L1 classical",
        "pass-through",
        True,
        False,
        False,
        "signal",
        "internal",
        "Direct low-grade IMU pass-through.",
    ),
    "bias_mean": BaselineMetadata(
        "bias_mean",
        "L1 classical",
        "bias calibration",
        True,
        False,
        True,
        "signal",
        "standard IMU bias calibration",
        "Subtracts train-set mean low-minus-high bias.",
    ),
    "affine": BaselineMetadata(
        "affine",
        "L1 classical",
        "affine calibration",
        True,
        False,
        True,
        "signal",
        "standard least-squares calibration",
        "Least-squares affine calibration fitted on the training split.",
    ),
    "ridge_affine": BaselineMetadata(
        "ridge_affine",
        "L1 classical",
        "regularized calibration",
        True,
        False,
        True,
        "signal",
        "standard ridge regression",
        "Ridge-regularized affine calibration for better conditioning.",
    ),
    "learned_fir_causal": BaselineMetadata(
        "learned_fir_causal",
        "L1 system ID",
        "learned FIR",
        True,
        False,
        True,
        "signal",
        "linear system identification",
        "Supervised causal multi-output FIR mapping from low-grade history to high-grade IMU.",
    ),
    "arx_linear": BaselineMetadata(
        "arx_linear",
        "L1 system ID",
        "ARX/NARX",
        True,
        False,
        True,
        "signal",
        "linear ARX system identification",
        "Causal lagged low-IMU and difference features with ridge fitting.",
    ),
    "frequency_transfer": BaselineMetadata(
        "frequency_transfer",
        "L1 system ID",
        "frequency response",
        False,
        True,
        True,
        "signal",
        "frequency-domain transfer-function fitting",
        "Learns per-channel complex frequency response from train data and applies it offline.",
    ),
    "moving_average_causal": BaselineMetadata(
        "moving_average_causal",
        "L1 filtering",
        "moving average",
        True,
        False,
        False,
        "signal",
        "standard causal smoothing",
        "Causal moving average.",
    ),
    "butterworth_causal": BaselineMetadata(
        "butterworth_causal",
        "L1 filtering",
        "Butterworth low-pass",
        True,
        False,
        False,
        "signal",
        "standard causal low-pass filtering",
        "Causal Butterworth low-pass filter.",
    ),
    "savgol_noncausal": BaselineMetadata(
        "savgol_noncausal",
        "L1 filtering",
        "Savitzky-Golay",
        False,
        True,
        False,
        "signal",
        "standard offline smoothing",
        "Non-causal Savitzky-Golay smoothing.",
    ),
    "haar_wavelet": BaselineMetadata(
        "haar_wavelet",
        "L1 filtering",
        "wavelet shrinkage",
        False,
        True,
        False,
        "signal",
        "wavelet denoising",
        "Offline Haar-wavelet shrinkage plus train-set bias correction.",
    ),
    "rts_smoother_affine": BaselineMetadata(
        "rts_smoother_affine",
        "L1 filtering",
        "offline smoother",
        False,
        True,
        True,
        "signal",
        "RTS/Kalman-smoother analogue",
        "Affine calibration followed by non-causal smoothing; labeled offline because it uses future samples.",
    ),
    "brossard_gyro_cnn": BaselineMetadata(
        "brossard_gyro_cnn",
        "L2 public neural signal",
        "dilated CNN gyro denoising",
        True,
        False,
        True,
        "signal",
        "https://github.com/mbrossar/denoise-imu-gyro",
        "Brossard-style dilated CNN gyro residual denoiser; accel is supplied by affine calibration.",
    ),
    "calibnet_cnn": BaselineMetadata(
        "calibnet_cnn",
        "L2 public neural signal",
        "Calib-Net-style CNN",
        True,
        False,
        True,
        "signal",
        "https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.772583/full",
        "Calib-Net-style supervised 1D CNN residual calibration for all six IMU channels.",
    ),
    "ronin_resnet_adapter": BaselineMetadata(
        "ronin_resnet_adapter",
        "L3 public navigation adapted",
        "RoNIN-style residual ResNet",
        True,
        False,
        True,
        "signal adapter",
        "https://ronin.cs.sfu.ca/",
        "RoNIN-inspired residual 1D ResNet adapted to signal-level virtual IMU output; not the original RoNIN task.",
    ),
}

NAVIGATION_ONLY_REFERENCES: dict[str, BaselineMetadata] = {
    "ronin_original": BaselineMetadata(
        "ronin_original",
        "L3 navigation-only reference",
        "neural inertial navigation",
        True,
        False,
        True,
        "trajectory",
        "https://ronin.cs.sfu.ca/",
        "Original RoNIN estimates motion/trajectory from IMU; not a high-grade IMU signal generator.",
    ),
    "tlio_original": BaselineMetadata(
        "tlio_original",
        "L3 navigation-only reference",
        "learned inertial odometry + EKF",
        True,
        False,
        True,
        "trajectory/filter state",
        "https://cathias.github.io/TLIO/",
        "TLIO fuses learned relative-state estimates and uncertainty in an EKF; not directly signal-level comparable.",
    ),
    "airimu_original": BaselineMetadata(
        "airimu_original",
        "L3 navigation-only reference",
        "uncertainty-aware inertial odometry",
        True,
        False,
        True,
        "trajectory/uncertainty",
        "https://github.com/haleqiu/AirIMU",
        "AirIMU targets uncertainty propagation for inertial odometry; useful for future uncertainty heads.",
    ),
    "airio_original": BaselineMetadata(
        "airio_original",
        "L3 navigation-only reference",
        "body-frame learned inertial odometry",
        True,
        False,
        True,
        "trajectory/filter state",
        "https://github.com/Air-IO/Air-IO",
        "AirIO targets UAV inertial odometry with body-frame representation and EKF fusion; compare via exported trajectory/state predictions.",
    ),
}


def _load_torch() -> tuple[Any, Any, Any, Any, Any]:
    """Load PyTorch only when neural baselines are requested."""
    try:
        import torch
        from torch import nn
        from torch.nn import functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - exercised only on broken torch installs.
        raise RuntimeError(
            "PyTorch is required for neural baselines. Set neural_epochs=0 for classical and external "
            "SOTA navigation evaluation, or fix the local torch installation."
        ) from exc
    return torch, nn, F, DataLoader, TensorDataset


def _make_dilated_cnn(in_channels: int, out_channels: int, hidden: int = 32) -> Any:
    """Create a small Brossard-style dilated CNN."""
    _, nn, _, _, _ = _load_torch()

    class DilatedCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, hidden, 5, padding=2, dilation=1),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, 5, padding=8, dilation=4),
                nn.ReLU(),
                nn.Conv1d(hidden, out_channels, 1),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    return DilatedCNN()


def _make_calibnet_cnn(in_channels: int = 6, out_channels: int = 6, hidden: int = 32) -> Any:
    """Create a compact Calib-Net-style CNN calibration model."""
    _, nn, _, _, _ = _load_torch()

    class CalibNetCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, hidden, 7, padding=3),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, 5, padding=2),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden, out_channels, 1),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    return CalibNetCNN()


def _make_resnet_signal_adapter(in_channels: int = 6, out_channels: int = 6, hidden: int = 32) -> Any:
    """Create a RoNIN-inspired residual 1D ResNet adapted to signal generation."""
    _, nn, functional, _, _ = _load_torch()

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int) -> None:
            super().__init__()
            padding = dilation
            self.block = nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
                nn.ReLU(),
                nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation),
            )
            self.activation = nn.ReLU()

        def forward(self, x: Any) -> Any:
            return self.activation(x + self.block(x))

    class ResNetSignalAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Conv1d(in_channels, hidden, 3, padding=1)
            self.blocks = nn.Sequential(ResidualBlock(hidden, 1), ResidualBlock(hidden, 2), ResidualBlock(hidden, 4))
            self.head = nn.Conv1d(hidden, out_channels, 1)

        def forward(self, x: Any) -> Any:
            return self.head(self.blocks(functional.relu(self.input(x))))

    return ResNetSignalAdapter()


def _sample_rate_hz(df: pd.DataFrame) -> float:
    diffs = df.sort_values("timestamp_s")["timestamp_s"].diff().dropna().to_numpy()
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(1.0 / np.median(diffs)) if diffs.size else 100.0


def _with_virtual(df: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    for idx, col in enumerate(VIRTUAL_COLUMNS):
        out[col] = values[:, idx]
    return out


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float = 1e-3, intercept: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if intercept:
        x_design = np.column_stack([x, np.ones(len(x))])
    else:
        x_design = x
    reg = alpha * np.eye(x_design.shape[1])
    if intercept:
        reg[-1, -1] = 0.0
    coef = np.linalg.solve(x_design.T @ x_design + reg, x_design.T @ y)
    if intercept:
        return coef[:-1], coef[-1]
    return coef, np.zeros(y.shape[1])


def _ridge_predict(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weights + bias


def _fit_affine(train_df: pd.DataFrame) -> AffineCalibration:
    x = train_df[LOW_COLUMNS].to_numpy(dtype=float)
    y = train_df[TARGET_COLUMNS].to_numpy(dtype=float)
    return AffineCalibration().fit(x, y)


def _lag_features(values: np.ndarray, taps: int, include_delta: bool = False) -> np.ndarray:
    padded = np.pad(values, ((taps - 1, 0), (0, 0)), mode="edge")
    features = [padded[taps - 1 - lag : taps - 1 - lag + len(values)] for lag in range(taps)]
    if include_delta:
        delta = np.diff(values, axis=0, prepend=values[[0]])
        features.append(delta)
    return np.concatenate(features, axis=1)


def _sequence_lag_features(
    df: pd.DataFrame,
    taps: int,
    include_delta: bool = False,
) -> np.ndarray:
    if "sequence_id" not in df.columns:
        return _lag_features(df[LOW_COLUMNS].to_numpy(dtype=float), taps, include_delta)
    if "timestamp_s" not in df.columns:
        raise ValueError("Data with sequence_id requires timestamp_s for sequence-aware lag features")

    feature_count = len(LOW_COLUMNS) * (taps + int(include_delta))
    features = np.empty((len(df), feature_count), dtype=float)
    groups = df.groupby("sequence_id", sort=False, dropna=False).indices.values()
    for positions in groups:
        timestamps = df.iloc[positions]["timestamp_s"].to_numpy(dtype=float)
        ordered_positions = positions[np.argsort(timestamps, kind="stable")]
        ordered_values = df.iloc[ordered_positions][LOW_COLUMNS].to_numpy(dtype=float)
        features[ordered_positions] = _lag_features(ordered_values, taps, include_delta)
    return features


def fit_system_id_baseline_state(
    train_df: pd.DataFrame,
    fir_taps: int = 9,
    alpha: float = 1e-2,
) -> dict[str, Any]:
    """Fit learned FIR and ARX baseline weights and return a serializable state."""
    train_missing = [col for col in [*LOW_COLUMNS, *TARGET_COLUMNS] if col not in train_df.columns]
    if train_missing:
        raise ValueError(f"System-ID baseline training data is missing columns: {train_missing}")
    ordered_train = train_df
    if "sequence_id" in train_df.columns:
        if "timestamp_s" not in train_df.columns:
            raise ValueError("Data with sequence_id requires timestamp_s for sequence-aware lag features")
        ordered_train = train_df.sort_values(["sequence_id", "timestamp_s"], kind="stable")
    train_high = ordered_train[TARGET_COLUMNS].to_numpy(dtype=float)
    fir_weights, fir_bias = _ridge_fit(
        _sequence_lag_features(ordered_train, taps=fir_taps),
        train_high,
        alpha=alpha,
    )
    arx_taps = max(fir_taps, 12)
    arx_weights, arx_bias = _ridge_fit(
        _sequence_lag_features(ordered_train, taps=arx_taps, include_delta=True),
        train_high,
        alpha=alpha,
    )
    return {
        "fir_taps": int(fir_taps),
        "arx_taps": int(arx_taps),
        "alpha": float(alpha),
        "fir_weights": fir_weights.tolist(),
        "fir_bias": fir_bias.tolist(),
        "arx_weights": arx_weights.tolist(),
        "arx_bias": arx_bias.tolist(),
    }


def predict_system_id_from_state(state: dict[str, Any], target_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Predict learned FIR and ARX baselines from a fitted serializable state."""
    target_missing = [col for col in LOW_COLUMNS if col not in target_df.columns]
    if target_missing:
        raise ValueError(f"System-ID baseline target data is missing columns: {target_missing}")
    fir_taps = int(state["fir_taps"])
    arx_taps = int(state["arx_taps"])
    return {
        "learned_fir_causal": _ridge_predict(
            _sequence_lag_features(target_df, taps=fir_taps),
            np.asarray(state["fir_weights"], dtype=float),
            np.asarray(state["fir_bias"], dtype=float),
        ),
        "arx_linear": _ridge_predict(
            _sequence_lag_features(target_df, taps=arx_taps, include_delta=True),
            np.asarray(state["arx_weights"], dtype=float),
            np.asarray(state["arx_bias"], dtype=float),
        ),
    }


def select_system_id_competitor(
    predictions: dict[str, np.ndarray],
    mode: str = "blend",
    blend_weight_fir: float = 0.5,
) -> np.ndarray:
    """Select or blend learned FIR and ARX predictions into one competitor."""
    mode_key = mode.lower().strip()
    if mode_key == "fir":
        return predictions["learned_fir_causal"]
    if mode_key == "arx":
        return predictions["arx_linear"]
    if mode_key == "blend":
        weight_fir = float(np.clip(blend_weight_fir, 0.0, 1.0))
        return weight_fir * predictions["learned_fir_causal"] + (1.0 - weight_fir) * predictions["arx_linear"]
    raise ValueError(f"Unsupported system-ID competitor mode: {mode}")


def predict_system_id_baselines(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    fir_taps: int = 9,
    alpha: float = 1e-2,
) -> dict[str, np.ndarray]:
    """Predict strong causal system-ID baselines for a target split.

    The returned methods are the same learned FIR and ARX baselines used in
    evaluation, but without any file I/O or neural baseline training. This
    makes them suitable as training-time competitors for the main model.
    """
    state = fit_system_id_baseline_state(train_df, fir_taps=fir_taps, alpha=alpha)
    return predict_system_id_from_state(state, target_df)


def add_system_id_competitor_columns(
    train_df: pd.DataFrame,
    target_df: pd.DataFrame,
    fir_taps: int = 9,
    mode: str = "blend",
    blend_weight_fir: float = 0.5,
    alpha: float = 1e-2,
    state: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add ARX/FIR competitor columns to a split for training-time margins.

    Parameters
    ----------
    train_df:
        Paired split used to fit the causal system-ID baselines.
    target_df:
        Split that receives `*_competitor` columns.
    mode:
        One of `blend`, `fir`, or `arx`. `blend` averages learned FIR and ARX
        predictions and is the default because it is less brittle than choosing
        one linear baseline a priori.
    """
    fitted = state or fit_system_id_baseline_state(train_df, fir_taps=fir_taps, alpha=alpha)
    predictions = predict_system_id_from_state(fitted, target_df)
    competitor = select_system_id_competitor(predictions, mode=mode, blend_weight_fir=blend_weight_fir)
    out = target_df.copy()
    for idx, col in enumerate(COMPETITOR_COLUMNS):
        out[col] = competitor[:, idx]
    return out


def _frequency_transfer_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    train_low = train_df[LOW_COLUMNS].to_numpy(float)
    train_high = train_df[TARGET_COLUMNS].to_numpy(float)
    test_low = test_df[LOW_COLUMNS].to_numpy(float)
    n_train = len(train_low)
    n_test = len(test_low)
    train_freq = np.fft.rfftfreq(n_train, d=1.0 / _sample_rate_hz(train_df))
    test_freq = np.fft.rfftfreq(n_test, d=1.0 / _sample_rate_hz(test_df))
    out = np.zeros_like(test_low)
    for channel in range(test_low.shape[1]):
        x_mean = np.mean(train_low[:, channel])
        y_mean = np.mean(train_high[:, channel])
        x_fft = np.fft.rfft(train_low[:, channel] - x_mean)
        y_fft = np.fft.rfft(train_high[:, channel] - y_mean)
        response = y_fft * np.conj(x_fft) / (np.abs(x_fft) ** 2 + 1e-6)
        response_abs = np.maximum(np.abs(response), 1e-12)
        response = np.where(response_abs > 4.0, 4.0 * response / response_abs, response)
        real = np.interp(test_freq, train_freq, response.real)
        imag = np.interp(test_freq, train_freq, response.imag)
        test_fft = np.fft.rfft(test_low[:, channel] - x_mean)
        out[:, channel] = np.fft.irfft(test_fft * (real + 1j * imag), n=n_test) + y_mean
    return out


def _haar_denoise_1d(x: np.ndarray, levels: int = 3, threshold_scale: float = 1.0) -> np.ndarray:
    n = len(x)
    target = 1 << int(np.ceil(np.log2(max(n, 2))))
    arr = np.pad(x.astype(float), (0, target - n), mode="edge")
    details = []
    approx = arr
    for _ in range(levels):
        if len(approx) < 2:
            break
        avg = (approx[0::2] + approx[1::2]) / np.sqrt(2.0)
        detail = (approx[0::2] - approx[1::2]) / np.sqrt(2.0)
        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745 if len(detail) else 0.0
        threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(max(len(detail), 2)))
        detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
        details.append(detail)
        approx = avg
    for detail in reversed(details):
        up = np.zeros(detail.size * 2)
        up[0::2] = (approx + detail) / np.sqrt(2.0)
        up[1::2] = (approx - detail) / np.sqrt(2.0)
        approx = up
    return approx[:n]


def _haar_wavelet(values: np.ndarray) -> np.ndarray:
    return np.column_stack([_haar_denoise_1d(values[:, axis]) for axis in range(values.shape[1])])


def _make_windows(values: np.ndarray, targets: np.ndarray, window_size: int, stride: int) -> tuple[Any, Any]:
    torch, _, _, _, _ = _load_torch()
    xs = []
    ys = []
    for start in range(0, len(values) - window_size + 1, stride):
        end = start + window_size
        xs.append(values[start:end].T)
        ys.append(targets[start:end].T)
    if not xs:
        raise ValueError("Not enough samples for neural baseline windows")
    return torch.tensor(np.stack(xs), dtype=torch.float32), torch.tensor(np.stack(ys), dtype=torch.float32)


def _fit_neural_residual(
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_channels: int,
    channel_slice: slice,
    epochs: int,
    window_size: int,
    stride: int,
    lr: float,
) -> np.ndarray:
    torch, _, functional, data_loader_cls, tensor_dataset_cls = _load_torch()
    train_low = train_df[LOW_COLUMNS].to_numpy(np.float32)
    train_high = train_df[TARGET_COLUMNS].to_numpy(np.float32)
    test_low = test_df[LOW_COLUMNS].to_numpy(np.float32)
    low_mean = train_low.mean(axis=0)
    low_std = np.where(train_low.std(axis=0) < 1e-6, 1.0, train_low.std(axis=0))
    residual = train_high[:, channel_slice] - train_low[:, channel_slice]
    residual_mean = residual.mean(axis=0)
    residual_std = np.where(residual.std(axis=0) < 1e-6, 1.0, residual.std(axis=0))
    x_train = (train_low - low_mean) / low_std
    y_train = (residual - residual_mean) / residual_std
    x_windows, y_windows = _make_windows(x_train, y_train, window_size=window_size, stride=stride)
    loader = data_loader_cls(tensor_dataset_cls(x_windows, y_windows), batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            if pred.shape[1] != out_channels:
                raise RuntimeError("Neural baseline output channel mismatch")
            loss = functional.smooth_l1_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    x_test = ((test_low - low_mean) / low_std).astype(np.float32)
    n = len(x_test)
    pred_sum = np.zeros((n, out_channels), dtype=float)
    pred_count = np.zeros((n, 1), dtype=float)
    starts = list(range(0, max(n - window_size + 1, 1), stride))
    if starts[-1] != max(n - window_size, 0):
        starts.append(max(n - window_size, 0))
    with torch.no_grad():
        for start in starts:
            end = min(start + window_size, n)
            window = x_test[start:end]
            pad = window_size - len(window)
            if pad > 0:
                window = np.pad(window, ((0, pad), (0, 0)), mode="edge")
            pred = model(torch.tensor(window.T[None], dtype=torch.float32)).numpy()[0].T[: end - start]
            pred = pred * residual_std + residual_mean
            pred_sum[start:end] += pred
            pred_count[start:end] += 1.0
    return pred_sum / np.maximum(pred_count, 1.0)


def _baseline_predictions(
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    moving_average_window: int,
    butterworth_cutoff_hz: float,
    savgol_window: int,
    fir_taps: int,
    neural_epochs: int,
    neural_window_size: int,
    neural_stride: int,
) -> dict[str, np.ndarray]:
    low_values = test_df[LOW_COLUMNS].to_numpy(dtype=float)
    train_low = train_df[LOW_COLUMNS].to_numpy(dtype=float)
    train_high = train_df[TARGET_COLUMNS].to_numpy(dtype=float)
    rate_hz = _sample_rate_hz(test_df)
    affine = _fit_affine(train_df)
    ridge_weights, ridge_bias = _ridge_fit(train_low, train_high, alpha=1e-2)
    bias = np.mean(train_low - train_high, axis=0)
    system_id_predictions = predict_system_id_baselines(
        train_df,
        test_df,
        fir_taps=fir_taps,
        alpha=1e-2,
    )
    rts_affine = savitzky_golay(affine.predict(low_values), window=savgol_window, polyorder=2)
    wavelet = _haar_wavelet(low_values - bias)

    predictions: dict[str, np.ndarray] = {
        "identity": low_values,
        "bias_mean": low_values - bias,
        "affine": affine.predict(low_values),
        "ridge_affine": _ridge_predict(low_values, ridge_weights, ridge_bias),
        "learned_fir_causal": system_id_predictions["learned_fir_causal"],
        "arx_linear": system_id_predictions["arx_linear"],
        "frequency_transfer": _frequency_transfer_predict(train_df, test_df),
        "moving_average_causal": moving_average(low_values, window=moving_average_window, causal=True),
        "butterworth_causal": butterworth_lowpass(low_values, rate_hz=rate_hz, cutoff_hz=butterworth_cutoff_hz, order=3),
        "savgol_noncausal": savitzky_golay(low_values, window=savgol_window, polyorder=2),
        "haar_wavelet": wavelet,
        "rts_smoother_affine": rts_affine,
    }

    if neural_epochs > 0:
        logger.info("Training neural baseline: brossard_gyro_cnn")
        gyro_residual = _fit_neural_residual(
            _make_dilated_cnn(len(LOW_COLUMNS), 3),
            train_df,
            test_df,
            out_channels=3,
            channel_slice=slice(0, 3),
            epochs=neural_epochs,
            window_size=neural_window_size,
            stride=neural_stride,
            lr=1e-3,
        )
        brossard = affine.predict(low_values)
        brossard[:, :3] = low_values[:, :3] + gyro_residual
        predictions["brossard_gyro_cnn"] = brossard

        logger.info("Training neural baseline: calibnet_cnn")
        calib_residual = _fit_neural_residual(
            _make_calibnet_cnn(len(LOW_COLUMNS), 6),
            train_df,
            test_df,
            out_channels=6,
            channel_slice=slice(0, 6),
            epochs=neural_epochs,
            window_size=neural_window_size,
            stride=neural_stride,
            lr=1e-3,
        )
        predictions["calibnet_cnn"] = low_values + calib_residual

        logger.info("Training neural baseline: ronin_resnet_adapter")
        ronin_residual = _fit_neural_residual(
            _make_resnet_signal_adapter(len(LOW_COLUMNS), 6),
            train_df,
            test_df,
            out_channels=6,
            channel_slice=slice(0, 6),
            epochs=neural_epochs,
            window_size=neural_window_size,
            stride=neural_stride,
            lr=1e-3,
        )
        predictions["ronin_resnet_adapter"] = low_values + ronin_residual

    return predictions


def _evaluate_virtual_frame(df: pd.DataFrame, min_allan_duration_s: float) -> dict[str, Any]:
    low = signal_metrics(df, "low")
    virtual = signal_metrics(df, "virtual")
    return {
        "signal": {"low": low, "virtual": virtual, "improvement": snr_improvement(low, virtual)},
        "spectral": {"low": spectral_metrics(df, "low"), "virtual": spectral_metrics(df, "virtual")},
        "allan": {
            "low": allan_summary(df, "low", min_duration_s=min_allan_duration_s),
            "virtual": allan_summary(df, "virtual", min_duration_s=min_allan_duration_s),
            "high": allan_summary(df, "high", min_duration_s=min_allan_duration_s),
        },
        "preintegration": {
            "low": preintegration_metrics(df, "low"),
            "virtual": preintegration_metrics(df, "virtual"),
        },
        "navigation": {"low": navigation_metrics(df, "low"), "virtual": navigation_metrics(df, "virtual")},
    }


def _leaderboard_key(metadata: dict[str, Any]) -> str:
    """Assign a method to the causal online or noncausal offline leaderboard."""
    if bool(metadata.get("uses_future", False)) or not bool(metadata.get("causal", True)):
        return "noncausal_offline"
    return "causal_online"


def _read_prediction_table(path: str | Path) -> pd.DataFrame:
    """Read an external SOTA inertial odometry prediction table."""
    pred_path = Path(path)
    suffix = pred_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(pred_path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(pred_path)
    if suffix == ".json":
        return pd.read_json(pred_path)
    raise ValueError(f"Unsupported navigation prediction format: {pred_path}")


def _first_existing_columns(df: pd.DataFrame, candidates: list[list[str]], kind: str) -> list[str]:
    for cols in candidates:
        if all(col in df.columns for col in cols):
            return cols
    raise ValueError(
        f"External navigation prediction is missing {kind} columns. "
        f"Expected one of: {candidates}. Available: {list(df.columns)}"
    )


def _align_external_series(pred_df: pd.DataFrame, ref_t: np.ndarray, cols: list[str]) -> np.ndarray:
    values = pred_df[cols].to_numpy(dtype=float)
    if "timestamp_s" not in pred_df.columns:
        if len(values) != len(ref_t):
            raise ValueError("External navigation prediction without timestamp_s must match test sample count")
        return values
    pred_t = pred_df["timestamp_s"].to_numpy(dtype=float)
    order = np.argsort(pred_t)
    pred_t = pred_t[order]
    values = values[order]
    return np.column_stack(
        [np.interp(ref_t, pred_t, values[:, axis], left=values[0, axis], right=values[-1, axis]) for axis in range(values.shape[1])]
    )


def _external_navigation_metrics(test_df: pd.DataFrame, pred_df: pd.DataFrame, source_path: str | Path) -> dict[str, Any]:
    """Evaluate official SOTA inertial odometry trajectory/state predictions."""
    ref_t = test_df["timestamp_s"].to_numpy(dtype=float)
    ref = integrate_dead_reckoning(test_df, "high")
    pos_cols = _first_existing_columns(
        pred_df,
        [
            ["position_x_m", "position_y_m", "position_z_m"],
            ["px", "py", "pz"],
            ["pos_x", "pos_y", "pos_z"],
            ["x", "y", "z"],
        ],
        "position",
    )
    pred_pos = _align_external_series(pred_df, ref_t, pos_cols)
    pos_err = np.linalg.norm(pred_pos - ref["position"], axis=1)
    metrics: dict[str, Any] = {
        "source_path": str(source_path),
        "samples": int(len(pred_pos)),
        "leaderboard": "causal_online",
        "output": "trajectory/state",
        "position_drift_m_final": float(pos_err[-1]) if len(pos_err) else float("nan"),
        "position_drift_m_rms": float(np.sqrt(np.nanmean(pos_err**2))),
        "position_drift_m_max": float(np.nanmax(pos_err)),
        "reference_note": "High-grade IMU integration is used as proxy reference unless external ground truth is supplied.",
    }
    velocity_candidates = [
        ["velocity_x_m_s", "velocity_y_m_s", "velocity_z_m_s"],
        ["vx", "vy", "vz"],
        ["vel_x", "vel_y", "vel_z"],
    ]
    attitude_candidates = [
        ["attitude_x_rad", "attitude_y_rad", "attitude_z_rad"],
        ["roll_rad", "pitch_rad", "yaw_rad"],
        ["rx", "ry", "rz"],
    ]
    for key, candidates, ref_key, final_name, rms_name in [
        ("velocity", velocity_candidates, "velocity", "velocity_drift_m_s_final", "velocity_drift_m_s_rms"),
        ("attitude", attitude_candidates, "attitude", "attitude_drift_rad_final", "attitude_drift_rad_rms"),
    ]:
        try:
            cols = _first_existing_columns(pred_df, candidates, key)
        except ValueError:
            continue
        pred_values = _align_external_series(pred_df, ref_t, cols)
        err = np.linalg.norm(pred_values - ref[ref_key], axis=1)
        metrics[final_name] = float(err[-1]) if len(err) else float("nan")
        metrics[rms_name] = float(np.sqrt(np.nanmean(err**2)))
    return metrics


def evaluate_sota_navigation_predictions(
    test_df: pd.DataFrame,
    prediction_paths: list[str | Path] | None,
) -> dict[str, Any]:
    """Evaluate official SOTA inertial odometry outputs exported by their own codebases."""
    external: dict[str, Any] = {}
    for path in prediction_paths or []:
        pred_df = _read_prediction_table(path)
        if "method" in pred_df.columns and pred_df["method"].nunique(dropna=True) == 1:
            name = str(pred_df["method"].dropna().iloc[0])
        else:
            name = Path(path).stem
        external[name] = {
            "metadata": {
                "name": name,
                "layer": "L3 SOTA inertial odometry",
                "family": "external official implementation",
                "causal": True,
                "uses_future": False,
                "trained": True,
                "output": "trajectory/state",
                "public_reference": "user-supplied official SOTA export",
                "note": "Evaluated from exported trajectory/state predictions, not adapted IMU-signal outputs.",
            },
            "navigation": _external_navigation_metrics(test_df, pred_df, path),
        }
    return {
        "references": {name: asdict(meta) for name, meta in NAVIGATION_ONLY_REFERENCES.items()},
        "external_predictions": external,
    }


def evaluate_baselines(
    input_path: str | Path,
    out_json: str | Path,
    train_path: str | Path | None = None,
    predictions_dir: str | Path | None = None,
    navigation_predictions: list[str | Path] | None = None,
    min_allan_duration_s: float = 20.0,
    moving_average_window: int = 5,
    butterworth_cutoff_hz: float = 12.0,
    savgol_window: int = 9,
    fir_taps: int = 9,
    neural_epochs: int = 4,
    neural_window_size: int = 128,
    neural_stride: int = 64,
) -> dict[str, Any]:
    """Evaluate a three-tier baseline suite on a paired IMU Parquet split."""
    if train_path is None:
        raise ValueError("train_path is required for fair baseline evaluation")
    test_df = pd.read_parquet(input_path)
    train_df = pd.read_parquet(train_path)
    missing = [col for col in [*LOW_COLUMNS, *TARGET_COLUMNS, "timestamp_s"] if col not in test_df.columns]
    if missing:
        raise ValueError(f"Baseline evaluation missing columns in test data: {missing}")
    train_missing = [col for col in [*LOW_COLUMNS, *TARGET_COLUMNS] if col not in train_df.columns]
    if train_missing:
        raise ValueError(f"Baseline evaluation missing columns in train data: {train_missing}")

    out_path = Path(out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_root = Path(predictions_dir) if predictions_dir else out_path.parent / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "input_path": str(input_path),
        "train_path": str(train_path) if train_path else str(input_path),
        "settings": {
            "moving_average_window": moving_average_window,
            "butterworth_cutoff_hz": butterworth_cutoff_hz,
            "savgol_window": savgol_window,
            "fir_taps": fir_taps,
            "neural_epochs": neural_epochs,
            "neural_window_size": neural_window_size,
            "neural_stride": neural_stride,
        },
        "baselines": {},
        "leaderboards": {
            "causal_online": {"baselines": [], "rows": []},
            "noncausal_offline": {"baselines": [], "rows": []},
        },
        "sota_inertial_odometry": evaluate_sota_navigation_predictions(test_df, navigation_predictions),
    }
    predictions = _baseline_predictions(
        test_df,
        train_df,
        moving_average_window=moving_average_window,
        butterworth_cutoff_hz=butterworth_cutoff_hz,
        savgol_window=savgol_window,
        fir_taps=fir_taps,
        neural_epochs=neural_epochs,
        neural_window_size=neural_window_size,
        neural_stride=neural_stride,
    )
    for name, values in predictions.items():
        frame = _with_virtual(test_df, values)
        pred_path = write_parquet(frame, pred_root / f"{name}.parquet")
        metrics = _evaluate_virtual_frame(frame, min_allan_duration_s=min_allan_duration_s)
        metadata = IMPLEMENTED_BASELINES[name]
        metrics["metadata"] = asdict(metadata)
        metrics["note"] = metadata.note
        metrics["prediction_path"] = str(pred_path)
        metrics["leaderboard"] = _leaderboard_key(asdict(metadata))
        result["baselines"][name] = metrics

    result["leaderboards"] = _make_leaderboards(result)

    out_path.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    write_baseline_csv(result, out_path.with_suffix(".csv"))
    write_baseline_markdown(result, out_path.with_suffix(".md"))
    write_sota_navigation_markdown(result["sota_inertial_odometry"], out_path.parent / "sota_navigation_metrics.md")
    write_sota_navigation_csv(result["sota_inertial_odometry"], out_path.parent / "sota_navigation_metrics.csv")
    try:
        from virtual_imu.viz.plots_baselines import plot_baseline_suite

        plot_baseline_suite(result, out_path.parent / "figures")
    except Exception as exc:
        logger.warning("Could not generate baseline figures: {}", exc)
    return result


def _baseline_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, metrics in result["baselines"].items():
        sig = metrics["signal"]["virtual"]
        spec = metrics["spectral"]["virtual"]
        preint = metrics["preintegration"]["virtual"]
        nav = metrics["navigation"]["virtual"]
        meta = metrics["metadata"]
        leaderboard = metrics.get("leaderboard", _leaderboard_key(meta))
        rows.append(
            {
                "baseline": name,
                "leaderboard": leaderboard,
                "layer": meta["layer"],
                "family": meta["family"],
                "causal": meta["causal"],
                "uses_future": meta["uses_future"],
                "trained": meta["trained"],
                "gyro_norm_rmse": sig.get("gyro_norm_rmse"),
                "accel_norm_rmse": sig.get("accel_norm_rmse"),
                "gyro_snr_db": sig.get("gyro_snr_db"),
                "accel_snr_db": sig.get("accel_snr_db"),
                "gyro_frequency_rmse": spec.get("gyro_frequency_band_rmse"),
                "gyro_coherence": spec.get("gyro_coherence"),
                "preintegration_residual": preint.get("imu_preintegration_residual"),
                "position_drift_m_final": nav.get("position_drift_m_final"),
                "prediction_path": metrics.get("prediction_path"),
            }
        )
    return rows


def _make_leaderboards(result: dict[str, Any]) -> dict[str, Any]:
    leaderboards: dict[str, Any] = {
        "causal_online": {"baselines": [], "rows": []},
        "noncausal_offline": {"baselines": [], "rows": []},
    }
    for row in _baseline_rows(result):
        key = row["leaderboard"]
        leaderboards.setdefault(key, {"baselines": [], "rows": []})
        leaderboards[key]["baselines"].append(row["baseline"])
        leaderboards[key]["rows"].append(row)
    for board in leaderboards.values():
        board["rows"] = sorted(board["rows"], key=lambda row: row["position_drift_m_final"])
    return leaderboards


def write_baseline_csv(result: dict[str, Any], path: str | Path) -> None:
    """Write compact baseline comparison CSV."""
    rows = _baseline_rows(result)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_baseline_markdown(result: dict[str, Any], path: str | Path) -> None:
    """Write compact baseline comparison Markdown."""
    lines = [
        "# Baseline Evaluation",
        "",
        f"- Input: `{result['input_path']}`",
        f"- Train: `{result['train_path']}`",
        "",
        "## Causal Online Leaderboard",
        "",
        "| Baseline | Layer | Causal | Trained | Gyro RMSE | Accel RMSE | Gyro Coh. | Preint Residual | Final Pos Drift [m] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.get("leaderboards", {}).get("causal_online", {}).get("rows", []):
        lines.append(
            f"| {row['baseline']} | {row['layer']} | {row['causal']} | {row['trained']} | "
            f"{row['gyro_norm_rmse']:.6g} | {row['accel_norm_rmse']:.6g} | "
            f"{row['gyro_coherence']:.3g} | {row['preintegration_residual']:.6g} | "
            f"{row['position_drift_m_final']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Noncausal Offline Leaderboard",
            "",
            "| Baseline | Layer | Causal | Trained | Gyro RMSE | Accel RMSE | Gyro Coh. | Preint Residual | Final Pos Drift [m] |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in result.get("leaderboards", {}).get("noncausal_offline", {}).get("rows", []):
        lines.append(
            f"| {row['baseline']} | {row['layer']} | {row['causal']} | {row['trained']} | "
            f"{row['gyro_norm_rmse']:.6g} | {row['accel_norm_rmse']:.6g} | "
            f"{row['gyro_coherence']:.3g} | {row['preintegration_residual']:.6g} | "
            f"{row['position_drift_m_final']:.6g} |"
        )
    lines.extend(["", "## Implemented Baseline Notes", ""])
    for name, metrics in result["baselines"].items():
        meta = metrics["metadata"]
        lines.append(f"- `{name}` ({meta['family']}): {metrics.get('note', '')}")
    lines.extend(
        [
            "",
            "## SOTA Inertial Odometry",
            "",
            "- Original RoNIN/TLIO/AirIMU/AirIO-style methods are evaluated through exported trajectory/state predictions in `sota_navigation_metrics.md`.",
            "- They are not treated as high-grade IMU signal generators.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sota_navigation_csv(result: dict[str, Any], path: str | Path) -> None:
    """Write compact external SOTA inertial odometry comparison CSV."""
    rows = []
    for name, metrics in result.get("external_predictions", {}).items():
        nav = metrics.get("navigation", {})
        rows.append(
            {
                "method": name,
                "leaderboard": nav.get("leaderboard", "causal_online"),
                "output": nav.get("output", "trajectory/state"),
                "position_drift_m_final": nav.get("position_drift_m_final"),
                "position_drift_m_rms": nav.get("position_drift_m_rms"),
                "source_path": nav.get("source_path"),
            }
        )
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method",
            "leaderboard",
            "output",
            "position_drift_m_final",
            "position_drift_m_rms",
            "source_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_sota_navigation_markdown(result: dict[str, Any], path: str | Path) -> None:
    """Write external SOTA inertial odometry comparison Markdown."""
    lines = [
        "# SOTA Inertial Odometry Evaluation",
        "",
        "This table evaluates trajectory/state predictions exported by official or external SOTA inertial odometry implementations.",
        "These methods are not direct high-grade IMU signal generators.",
        "",
        "## External Prediction Leaderboard",
        "",
        "| Method | Output | Final Pos Drift [m] | RMS Pos Drift [m] | Source |",
        "|---|---|---:|---:|---|",
    ]
    external = result.get("external_predictions", {})
    if external:
        for name, metrics in external.items():
            nav = metrics.get("navigation", {})
            lines.append(
                f"| {name} | {nav.get('output', 'trajectory/state')} | "
                f"{nav.get('position_drift_m_final', float('nan')):.6g} | "
                f"{nav.get('position_drift_m_rms', float('nan')):.6g} | "
                f"`{nav.get('source_path', '')}` |"
            )
    else:
        lines.append("| _none supplied_ |  |  |  |  |")
    lines.extend(["", "## Reference Methods", ""])
    lines.append("| Method | Output | Reference | Note |")
    lines.append("|---|---|---|---|")
    for name, meta in result.get("references", {}).items():
        lines.append(f"| {name} | {meta['output']} | {meta['public_reference']} | {meta['note']} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
