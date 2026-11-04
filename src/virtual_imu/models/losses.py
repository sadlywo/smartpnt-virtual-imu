"""Training losses for virtual IMU generation."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def signal_loss(pred: torch.Tensor, target: torch.Tensor, kind: str = "huber") -> torch.Tensor:
    """Huber or MSE signal loss."""
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "huber":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"Unsupported loss kind: {kind}")


def _loss_by_kind(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "huber":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"Unsupported loss kind: {kind}")


def split_sensor_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    gyro_weight: float = 1.0,
    accel_weight: float = 1.0,
) -> torch.Tensor:
    """Compute a gyro/accel weighted loss for the first 3 and last 3 channels."""
    gyro = _loss_by_kind(pred[:, :3], target[:, :3], kind)
    accel = _loss_by_kind(pred[:, 3:], target[:, 3:], kind)
    denom = max(gyro_weight + accel_weight, 1e-12)
    return (gyro_weight * gyro + accel_weight * accel) / denom


def derivative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match first temporal derivative."""
    return F.smooth_l1_loss(pred[..., 1:] - pred[..., :-1], target[..., 1:] - target[..., :-1])


def split_derivative_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gyro_weight: float = 1.0,
    accel_weight: float = 1.0,
) -> torch.Tensor:
    """Match first temporal derivative with separate gyro/accel emphasis."""
    pred_diff = pred[..., 1:] - pred[..., :-1]
    target_diff = target[..., 1:] - target[..., :-1]
    gyro = F.smooth_l1_loss(pred_diff[:, :3], target_diff[:, :3])
    accel = F.smooth_l1_loss(pred_diff[:, 3:], target_diff[:, 3:])
    denom = max(gyro_weight + accel_weight, 1e-12)
    return (gyro_weight * gyro + accel_weight * accel) / denom


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match coarse normalized FFT magnitude."""
    pred_fft = torch.fft.rfft(pred, dim=-1).abs()
    target_fft = torch.fft.rfft(target, dim=-1).abs()
    return F.l1_loss(torch.log1p(pred_fft), torch.log1p(target_fft))


def split_spectral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gyro_weight: float = 1.0,
    accel_weight: float = 1.0,
) -> torch.Tensor:
    """Match FFT magnitudes with separate gyro/accel emphasis."""
    pred_fft = torch.fft.rfft(pred, dim=-1).abs()
    target_fft = torch.fft.rfft(target, dim=-1).abs()
    gyro = F.l1_loss(torch.log1p(pred_fft[:, :3]), torch.log1p(target_fft[:, :3]))
    accel = F.l1_loss(torch.log1p(pred_fft[:, 3:]), torch.log1p(target_fft[:, 3:]))
    denom = max(gyro_weight + accel_weight, 1e-12)
    return (gyro_weight * gyro + accel_weight * accel) / denom


def allan_multiscale_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Lightweight Allan-inspired loss on multi-scale averages."""
    losses = []
    for scale in (2, 4, 8, 16):
        if pred.shape[-1] < scale * 2:
            continue
        pred_avg = F.avg_pool1d(pred, kernel_size=scale, stride=scale)
        target_avg = F.avg_pool1d(target, kernel_size=scale, stride=scale)
        losses.append(F.l1_loss(pred_avg[..., 1:] - pred_avg[..., :-1], target_avg[..., 1:] - target_avg[..., :-1]))
    return sum(losses) / max(len(losses), 1) if losses else pred.new_tensor(0.0)


def preintegration_consistency_loss(pred: torch.Tensor, target: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
    """Match integrated gyro and accel increments over each window."""
    if torch.are_deterministic_algorithms_enabled():
        # Torch 2.4 CUDA cumsum has no strict deterministic kernel. For the
        # short training windows, this fixed triangular linear map is the same
        # prefix sum and Jacobian, using deterministic cuBLAS when configured.
        prefix = torch.ones(pred.shape[-1], pred.shape[-1], device=pred.device, dtype=pred.dtype).triu()
        pred_delta = (pred @ prefix) * dt
        target_delta = (target @ prefix) * dt
    else:
        pred_delta = torch.cumsum(pred, dim=-1) * dt
        target_delta = torch.cumsum(target, dim=-1) * dt
    return F.smooth_l1_loss(pred_delta, target_delta)


def residual_highpass_loss(pred: torch.Tensor, cutoff_fraction: float = 0.35) -> torch.Tensor:
    """Penalize high-frequency gyro residual energy in residual-learning mode."""
    gyro = pred[:, :3]
    spectrum = torch.fft.rfft(gyro, dim=-1).abs()
    n_freq = spectrum.shape[-1]
    if n_freq < 4:
        return pred.new_tensor(0.0)
    cutoff = min(max(int(round(cutoff_fraction * n_freq)), 1), n_freq - 1)
    high = spectrum[..., cutoff:]
    return torch.mean(torch.log1p(high))


def baseline_margin_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    competitor: torch.Tensor,
    gyro_margin: float = 0.05,
    accel_margin: float = 0.02,
    gyro_weight: float = 3.0,
    accel_weight: float = 1.0,
) -> torch.Tensor:
    """Penalize predictions that do not beat the ARX/FIR competitor.

    The tensors must already live in the same target domain. In residual mode,
    for example, all three are residual signals rather than absolute IMU
    signals. The margin is relative: a `gyro_margin` of 0.05 requires the model
    gyro norm error to be at least 5% smaller than the competitor's norm error.
    """
    pred_gyro_err = torch.linalg.vector_norm(pred[:, :3] - target[:, :3], dim=1)
    comp_gyro_err = torch.linalg.vector_norm(competitor[:, :3] - target[:, :3], dim=1)
    pred_accel_err = torch.linalg.vector_norm(pred[:, 3:] - target[:, 3:], dim=1)
    comp_accel_err = torch.linalg.vector_norm(competitor[:, 3:] - target[:, 3:], dim=1)
    gyro = torch.relu(pred_gyro_err - (1.0 - gyro_margin) * comp_gyro_err).mean()
    accel = torch.relu(pred_accel_err - (1.0 - accel_margin) * comp_accel_err).mean()
    denom = max(gyro_weight + accel_weight, 1e-12)
    return (gyro_weight * gyro + accel_weight * accel) / denom


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cfg: dict,
    dt: float = 0.01,
    competitor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted training loss and scalar components."""
    loss_kind = str(cfg.get("loss", "huber"))
    base = split_sensor_loss(
        pred,
        target,
        loss_kind,
        gyro_weight=float(cfg.get("gyro_signal_weight", cfg.get("gyro_loss_weight", 1.0))),
        accel_weight=float(cfg.get("accel_signal_weight", cfg.get("accel_loss_weight", 1.0))),
    )
    d = split_derivative_loss(
        pred,
        target,
        gyro_weight=float(cfg.get("gyro_derivative_weight", 1.0)),
        accel_weight=float(cfg.get("accel_derivative_weight", 1.0)),
    ) * float(cfg.get("derivative_weight", 0.0))
    s = split_spectral_loss(
        pred,
        target,
        gyro_weight=float(cfg.get("gyro_spectral_weight", 1.0)),
        accel_weight=float(cfg.get("accel_spectral_weight", 1.0)),
    ) * float(cfg.get("spectral_weight", 0.0))
    a = allan_multiscale_loss(pred, target) * float(cfg.get("allan_weight", 0.0))
    p = preintegration_consistency_loss(pred, target, dt=dt) * float(cfg.get("preintegration_weight", 0.0))
    residual_hp = pred.new_tensor(0.0)
    if str(cfg.get("target_mode", "direct")) in {"residual", "competitor_residual"}:
        residual_hp = residual_highpass_loss(
            pred,
            cutoff_fraction=float(cfg.get("residual_highpass_cutoff_fraction", 0.35)),
        ) * float(cfg.get("residual_highpass_weight", 0.0))
    correction_energy = pred.new_tensor(0.0)
    if str(cfg.get("target_mode", "direct")) == "competitor_residual":
        correction_energy = split_sensor_loss(
            pred,
            torch.zeros_like(pred),
            "mse",
            gyro_weight=float(cfg.get("gyro_correction_energy_weight", 3.0)),
            accel_weight=float(cfg.get("accel_correction_energy_weight", 1.0)),
        ) * float(cfg.get("correction_energy_weight", 0.0))
    baseline_margin = pred.new_tensor(0.0)
    if competitor is not None:
        baseline_margin = baseline_margin_loss(
            pred,
            target,
            competitor,
            gyro_margin=float(cfg.get("gyro_baseline_margin", 0.05)),
            accel_margin=float(cfg.get("accel_baseline_margin", 0.02)),
            gyro_weight=float(cfg.get("gyro_baseline_margin_weight", 3.0)),
            accel_weight=float(cfg.get("accel_baseline_margin_weight", 1.0)),
        ) * float(cfg.get("baseline_margin_weight", 0.0))
    total = base + d + s + a + p + residual_hp + correction_energy + baseline_margin
    return total, {
        "signal": float(base.detach().cpu()),
        "derivative": float(d.detach().cpu()),
        "spectral": float(s.detach().cpu()),
        "allan": float(a.detach().cpu()),
        "preintegration": float(p.detach().cpu()),
        "residual_highpass": float(residual_hp.detach().cpu()),
        "correction_energy": float(correction_energy.detach().cpu()),
        "baseline_margin": float(baseline_margin.detach().cpu()),
    }
