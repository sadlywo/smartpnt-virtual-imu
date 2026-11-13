"""Physical-unit comparisons on identical, overlap-stitched held-out samples."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from virtual_imu.diffusion.data import PairedWindows

AXES = ("gx", "gy", "gz", "ax", "ay", "az")


def comparison_table(
    low: np.ndarray, high: np.ndarray, predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    groups = [(axis, [i], "rad/s" if i < 3 else "m/s^2") for i, axis in enumerate(AXES)]
    groups += [("gyro", [0, 1, 2], "rad/s"), ("accel", [3, 4, 5], "m/s^2")]
    for name, indexes, unit in groups:
        reference = high[:, indexes]
        low_error = low[:, indexes] - reference
        baseline_rmse = np.sqrt(np.mean(low_error**2))
        baseline_mae = np.mean(np.abs(low_error))
        baseline_power = np.mean(low_error**2)
        signal_power = np.mean(reference**2)
        for method, pred in predictions.items():
            error = pred[:, indexes] - reference
            mse = np.mean(error**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(error))
            rows.append(
                {
                    "component": name,
                    "method": method,
                    "unit": unit,
                    "samples": len(high),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "bias": float(np.mean(error))
                    if len(indexes) == 1
                    else float(np.sqrt(np.mean(np.mean(error, axis=0) ** 2))),
                    "bias_definition": "signed_axis_mean" if len(indexes) == 1 else "rms_axis_mean",
                    "noise_std": float(np.sqrt(np.mean(np.var(error, axis=0)))),
                    "rmse_improvement_pct": float(100 * (1 - rmse / baseline_rmse))
                    if baseline_rmse > 0
                    else float("nan"),
                    "mae_improvement_pct": float(100 * (1 - mae / baseline_mae))
                    if baseline_mae > 0
                    else float("nan"),
                    "snr_db": float(10 * np.log10(max(signal_power, 1e-30) / max(mse, 1e-30))),
                    "snr_gain_db": float(
                        10 * np.log10(max(baseline_power, 1e-30) / max(mse, 1e-30))
                    ),
                }
            )
    return pd.DataFrame(rows)


def markdown_table(table: pd.DataFrame) -> str:
    """Render without optional tabulate dependency."""
    columns = list(table.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in table.itertuples(index=False, name=None):
        cells = [f"{v:.6g}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


@torch.no_grad()
def sample_windows(
    model, dataset, device, batch_size: int, seed: int, draws: int = 1, amp: bool = False
) -> np.ndarray:
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    outputs = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for condition, _ in loader:
        condition = condition.to(device)
        # Independent Monte Carlo chains share a batched forward pass at each timestep.
        # This preserves the estimator while avoiding draws * T serial tiny CUDA launches.
        expanded = condition.repeat(draws, 1, 1)
        with torch.autocast(device_type=device.type, enabled=amp):
            samples = model.sample(expanded, generator=generator)
        average = samples.reshape(draws, len(condition), 6, condition.shape[-1]).mean(0)
        outputs.append(average.cpu().numpy())
    return np.concatenate(outputs)


def physical_prediction(
    model,
    windows: PairedWindows,
    device,
    batch_size: int,
    seed: int,
    draws: int = 1,
    amp: bool = False,
    window_indices=None,
):
    dataset = windows if window_indices is None else Subset(windows, list(window_indices))
    normalized = sample_windows(model, dataset, device, batch_size, seed, draws, amp)
    mean = np.array(windows.stats["residual_mean"])[None, :, None]
    std = np.array(windows.stats["residual_std"])[None, :, None]
    correction, mask = windows.stitch(normalized * std + mean, indexes=window_indices)
    prediction = windows.low + correction
    return prediction, mask


def relative_score(pred: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Equal weight to gyro and accel; never mix their physical units."""
    scores = []
    for slc in (slice(0, 3), slice(3, 6)):
        numerator = np.mean((pred[:, slc] - high[:, slc]) ** 2)
        denominator = np.mean((low[:, slc] - high[:, slc]) ** 2)
        scores.append(np.sqrt(numerator / max(denominator, 1e-20)))
    return float(np.mean(scores))
