"""DDPM tuning and improvement figures using the repository paper style."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from virtual_imu.viz.style import COLORS, apply_paper_style, save_figure


def plot_diffusion_results(run_dir, frame, metrics, search, rate_hz):
    apply_paper_style()
    out = run_dir / "figures"
    captions = []
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for trial in search:
        history = pd.read_csv(run_dir / "trials" / trial["trial"] / "history.csv")
        axs[0].plot(history.epoch, history.train_loss, label=trial["trial"])
        axs[1].plot(history.epoch, history.validation_score, label=trial["trial"])
    axs[0].set(xlabel="Epoch", ylabel="DDPM training objective")
    axs[1].set(xlabel="Epoch", ylabel="Validation relative RMSE")
    axs[1].axhline(1, color=COLORS["low"], linestyle="--", linewidth=0.8)
    axs[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save_figure(fig, out, "training_search")
    captions.append(
        (
            "training_search",
            "Training objective and fixed-seed full-DDPM validation "
            "score for every hyperparameter candidate. Lower validation scores are better; "
            "1.0 is the uncorrected input. Test data does not participate in selection.",
        )
    )
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for i, ax in enumerate(axs):
        group, unit = (("gyro", "rad/s"), ("accel", "m/s^2"))[i]
        rows = metrics[metrics.component.eq(group)]
        ax.bar(rows.method, rows.rmse, color=[COLORS["low"], COLORS["accent"], COLORS["virtual"]])
        ax.set(ylabel=f"{group.capitalize()} component RMSE [{unit}]")
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    save_figure(fig, out, "heldout_rmse")
    captions.append(
        (
            "heldout_rmse",
            "Held-out six-axis correction compared with raw input and "
            "a train-only affine calibration on identical overlap-stitched sample support. "
            "Gyro and acceleration RMSE are computed separately in physical units.",
        )
    )
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    axes = ["gx", "gy", "gz", "ax", "ay", "az"]
    for i, (method, color) in enumerate(
        (("affine", COLORS["accent"]), ("ddpm", COLORS["virtual"]))
    ):
        rows = metrics[metrics.method.eq(method)].set_index("component").loc[axes]
        ax.bar(
            np.arange(6) + (i - 0.5) * 0.35,
            rows.rmse_improvement_pct,
            width=0.35,
            label=method,
            color=color,
        )
    ax.set_xticks(np.arange(6), axes)
    ax.axhline(0, color=COLORS["high"], linewidth=0.8)
    ax.set_ylabel("RMSE reduction versus raw input [%]")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out, "axis_improvement")
    captions.append(
        (
            "axis_improvement",
            "Per-axis held-out RMSE reduction, 100(1 - model/raw). "
            "Negative values indicate degradation and are retained.",
        )
    )
    # Keep a continuous single recording for illustrative signal and PSD figures.
    group = max((g for _, g in frame.groupby("sequence_id", sort=False)), key=len)
    segment = group.iloc[: min(len(group), int(rate_hz * 30))]
    for sensor, axes, unit in (
        ("gyro", ["gx", "gy", "gz"], "rad/s"),
        ("accel", ["ax", "ay", "az"], "m/s^2"),
    ):
        t = segment.timestamp_s.to_numpy() - segment.timestamp_s.iloc[0]
        fig, axs = plt.subplots(3, 1, figsize=(7.2, 5.0), sharex=True)
        for i, ax in enumerate(axs):
            axis = axes[i]
            for prefix, color in (
                ("low", COLORS["low"]),
                ("virtual", COLORS["virtual"]),
                ("high", COLORS["high"]),
            ):
                ax.plot(t, segment[f"{axis}_{prefix}"], color=color, label=prefix)
            ax.set_ylabel(f"{axis} [{unit}]")
        axs[0].legend(frameon=False, ncol=3, loc="upper right")
        axs[-1].set_xlabel("Time [s]")
        fig.tight_layout()
        stem = f"imu_time_series_{sensor}"
        save_figure(fig, out, stem)
        captions.append(
            (
                stem,
                f"{sensor} signals in the first 30 s of the longest held-out "
                "recording. Low input, DDPM virtual IMU and measured VN100 reference.",
            )
        )
        fig, ax = plt.subplots(figsize=(7.2, 2.8))
        high = segment[[f"{a}_high" for a in axes]].to_numpy()
        for prefix, color in (("low", COLORS["low"]), ("virtual", COLORS["virtual"])):
            values = segment[[f"{a}_{prefix}" for a in axes]].to_numpy()
            ax.plot(t, np.linalg.norm(values - high, axis=1), color=color, label=prefix)
        ax.set(xlabel="Time [s]", ylabel=f"{sensor.capitalize()} error norm [{unit}]")
        ax.legend(frameon=False)
        fig.tight_layout()
        stem = f"{sensor}_error_comparison"
        save_figure(fig, out, stem)
        captions.append(
            (stem, f"{sensor} error norm against measured VN100 in the same held-out segment.")
        )
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for i, ax in enumerate(axs):
        axis, unit = (("gz", "(rad/s)^2/Hz"), ("az", "(m/s^2)^2/Hz"))[i]
        for method, color in (("low", COLORS["low"]), ("virtual", COLORS["virtual"])):
            error = group[f"{axis}_{method}"].to_numpy() - group[f"{axis}_high"].to_numpy()
            f, power = welch(error, fs=rate_hz, nperseg=min(256, len(error)))
            ax.semilogy(f[1:], np.maximum(power[1:], 1e-30), label=method, color=color)
        ax.set(
            xlabel="Frequency [Hz]", ylabel=f"{axis} residual PSD [{unit}]", xlim=(0, rate_hz / 2)
        )
        ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out, "residual_psd")
    captions.append(
        (
            "residual_psd",
            "Residual power spectral density for gz and az in the longest "
            "held-out sequence, comparing raw and DDPM signals against measured VN100.",
        )
    )
    lines = [
        "# Figure index",
        "",
        f"Run: {run_dir.resolve()}",
        "",
        "All figures are exported as PNG (300 dpi), SVG and PDF.",
        "",
    ]
    for stem, caption in captions:
        lines += [f"- [{stem}.png]({stem}.png): {caption}"]
    (out / "figure_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_diffusion_aggregate(table, out):
    apply_paper_style()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for i, ax in enumerate(axs):
        component = ("gyro", "accel")[i]
        rows = table[table.component.eq(component) & table.method.eq("ddpm")]
        ax.bar(
            rows.pair_id.str.replace("_to_VN100", "", regex=False),
            rows.rmse_improvement_pct,
            color=COLORS["virtual"],
        )
        ax.axhline(0, color=COLORS["high"], linewidth=0.8)
        ax.set_ylabel(f"{component.capitalize()} RMSE reduction [%]")
    fig.tight_layout()
    save_figure(fig, out, "three_sensor_improvement")
    (out / "figure_index.md").write_text(
        "# Figure index\n\n- three_sensor_improvement (PNG/SVG/PDF): Held-out component RMSE "
        "reduction for three independently trained ALiMU to VN100 DDPM models. "
        "Positive values show improvement; negative values show degradation.\n",
        encoding="utf-8",
    )
