from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.viz.style import COLORS, SIZES, apply_paper_style, save_figure

ALIGN_COLUMNS = [
    "timestamp_s",
    "gx_radps",
    "gy_radps",
    "gz_radps",
    "ax_mps2",
    "ay_mps2",
    "az_mps2",
    "gt_x_local_m",
    "gt_y_local_m",
    "gt_z_local_m",
    "gt_v_east_mps",
    "gt_v_north_mps",
    "gt_v_up_mps",
    "gt_heading_deg",
    "gt_pitch_deg",
    "gt_roll_deg",
    "gt_quality",
    "gt_ambiguity_status",
    "gt_ins_flag",
    "sequence_id",
]


def _downsample(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    step = int(np.ceil(len(frame) / max_points))
    return frame.iloc[::step].copy()


def _content_hash(frame: pd.DataFrame) -> str:
    comparable = frame.drop(columns=["sequence_id"], errors="ignore")
    hashed = pd.util.hash_pandas_object(comparable, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def _summarize_frame(path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    frame = pd.read_parquet(path)
    missing = [column for column in ALIGN_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    frame = frame[ALIGN_COLUMNS].sort_values("timestamp_s").reset_index(drop=True)
    seq = str(frame["sequence_id"].iloc[0]) if len(frame) else path.stem
    timestamps = frame["timestamp_s"].to_numpy(float)
    dt = np.diff(timestamps)
    accel = frame[["ax_mps2", "ay_mps2", "az_mps2"]].to_numpy(float)
    gyro = frame[["gx_radps", "gy_radps", "gz_radps"]].to_numpy(float)
    velocity = frame[["gt_v_east_mps", "gt_v_north_mps", "gt_v_up_mps"]].to_numpy(float)
    accel_norm = np.linalg.norm(accel, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    numeric = frame.select_dtypes(include=[np.number])
    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0

    summary = {
        "sequence_id": seq,
        "source": str(path),
        "rows": int(len(frame)),
        "start_s": float(timestamps[0]) if len(timestamps) else None,
        "end_s": float(timestamps[-1]) if len(timestamps) else None,
        "duration_s": duration,
        "rate_hz_est": float((len(frame) - 1) / duration) if duration > 0 else None,
        "median_dt_s": float(np.median(dt)) if len(dt) else None,
        "p95_dt_s": float(np.percentile(dt, 95)) if len(dt) else None,
        "max_dt_s": float(np.max(dt)) if len(dt) else None,
        "is_monotonic": bool(np.all(dt > 0)) if len(dt) else True,
        "nan_count_numeric": int(numeric.isna().sum().sum()),
        "accel_norm_median": float(np.median(accel_norm)),
        "accel_norm_p95": float(np.percentile(accel_norm, 95)),
        "accel_norm_max": float(np.max(accel_norm)),
        "gyro_norm_median": float(np.median(gyro_norm)),
        "gyro_norm_p95": float(np.percentile(gyro_norm, 95)),
        "gyro_norm_max": float(np.max(gyro_norm)),
        "speed_median_mps": float(np.median(speed)),
        "speed_p95_mps": float(np.percentile(speed, 95)),
        "span_x_m": float(frame["gt_x_local_m"].max() - frame["gt_x_local_m"].min()),
        "span_y_m": float(frame["gt_y_local_m"].max() - frame["gt_y_local_m"].min()),
        "span_z_m": float(frame["gt_z_local_m"].max() - frame["gt_z_local_m"].min()),
        "quality_counts": {str(k): int(v) for k, v in frame["gt_quality"].value_counts().to_dict().items()},
        "ins_flag_counts": {str(k): int(v) for k, v in frame["gt_ins_flag"].value_counts().to_dict().items()},
        "content_hash_excluding_sequence": _content_hash(frame),
    }
    return summary, frame


def _find_duplicate_content(summaries: list[dict[str, Any]]) -> list[list[str]]:
    groups: dict[str, list[str]] = {}
    for item in summaries:
        groups.setdefault(str(item["content_hash_excluding_sequence"]), []).append(str(item["sequence_id"]))
    return [seqs for seqs in groups.values() if len(seqs) > 1]


def _plot_trajectories(frames: dict[str, pd.DataFrame], out_dir: Path) -> list[Path]:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=SIZES["double"])
    palette = [COLORS["high"], COLORS["low"], COLORS["virtual"], COLORS["accent"]]
    for idx, (seq, frame) in enumerate(frames.items()):
        small = _downsample(frame, 14000)
        ax.plot(
            small["gt_x_local_m"],
            small["gt_y_local_m"],
            label=seq,
            linewidth=1.0,
            color=palette[idx % len(palette)],
        )
    ax.set_xlabel("Local X [m]")
    ax.set_ylabel("Local Y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return save_figure(fig, out_dir, "smartpnt_gt_trajectory_overview")


def _plot_signal_sanity(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    apply_paper_style()
    seq = str(frame["sequence_id"].iloc[0])
    t0 = float(frame["timestamp_s"].iloc[0])
    snippet = frame[frame["timestamp_s"] <= t0 + 120.0].copy()
    if snippet.empty:
        snippet = frame.head(min(len(frame), 12000)).copy()
    time_s = snippet["timestamp_s"].to_numpy(float) - float(snippet["timestamp_s"].iloc[0])
    gyro_norm = np.linalg.norm(snippet[["gx_radps", "gy_radps", "gz_radps"]].to_numpy(float), axis=1)
    speed = np.linalg.norm(
        snippet[["gt_v_east_mps", "gt_v_north_mps", "gt_v_up_mps"]].to_numpy(float), axis=1
    )

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=SIZES["tall"])
    axes[0].plot(time_s, snippet["ax_mps2"], label="ax", color=COLORS["low"])
    axes[0].plot(time_s, snippet["ay_mps2"], label="ay", color=COLORS["virtual"])
    axes[0].plot(time_s, snippet["az_mps2"], label="az", color=COLORS["accent"])
    axes[0].set_ylabel("Specific force [m/s^2]")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    axes[1].plot(time_s, gyro_norm, color=COLORS["high"])
    axes[1].set_ylabel("Angular rate [rad/s]")
    axes[2].plot(time_s, speed, color=COLORS["virtual"])
    axes[2].set_ylabel("GT speed [m/s]")
    axes[2].set_xlabel("Time [s]")
    fig.align_ylabels(axes)
    return save_figure(fig, out_dir, f"smartpnt_signal_sanity_{seq}")


def _plot_alignment_summary(summaries: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    apply_paper_style()
    labels = [str(item["sequence_id"]) for item in summaries]
    accel = [float(item["accel_norm_median"]) for item in summaries]
    gyro = [float(item["gyro_norm_median"]) for item in summaries]
    speed = [float(item["speed_median_mps"]) for item in summaries]
    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=SIZES["double"])
    ax.bar(x - width, accel, width=width, label="accel norm median", color=COLORS["low"])
    ax.bar(x, gyro, width=width, label="gyro norm median", color=COLORS["high"])
    ax.bar(x + width, speed, width=width, label="GT speed median", color=COLORS["virtual"])
    ax.set_xticks(x, labels)
    ax.set_yscale("log")
    ax.set_ylabel("Magnitude (log scale)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return save_figure(fig, out_dir, "smartpnt_alignment_summary")


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = ["# SmartPNT-MSF Alignment Quality", ""]
    lines.append(f"- Processed root: `{summary['processed_root']}`")
    lines.append(f"- Figure root: `{summary['figure_root']}`")
    lines.append("")
    lines.append("| sequence | rows | duration s | rate Hz | dt median s | NaNs | accel median | gyro median | speed p95 m/s | monotonic |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for item in summary["sequences"]:
        lines.append(
            f"| {item['sequence_id']} | {item['rows']} | {item['duration_s']:.3f} | "
            f"{item['rate_hz_est']:.3f} | {item['median_dt_s']:.5f} | "
            f"{item['nan_count_numeric']} | {item['accel_norm_median']:.6g} | "
            f"{item['gyro_norm_median']:.6g} | {item['speed_p95_mps']:.3f} | {item['is_monotonic']} |"
        )
    lines.append("")
    lines.append("## Duplicate Content Check")
    lines.append("")
    if summary["duplicate_sequence_groups"]:
        for group in summary["duplicate_sequence_groups"]:
            lines.append(f"- Numeric/string aligned content is identical apart from `sequence_id`: {', '.join(group)}")
    else:
        lines.append("- No duplicate aligned sequence content detected.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- The aligned files use GT timestamps as the output clock; IMU channels are linearly interpolated.")
    lines.append("- The aligned `gx_radps` and `ax_mps2` columns are expected to be physical rates after SmartPNT-MSF raw IMU increment conversion.")
    lines.append("- Keep the IMU field convention traceable: raw increments remain in `data/processed/smartpnt_msf/imu_raw`, while aligned files expose training-ready rate columns.")
    lines.append("- The current outputs are suitable for building a GT-supervised data loader, with the upstream ROS driver unit convention still worth documenting explicitly.")
    return "\n".join(lines) + "\n"


def _figure_index(figure_paths: list[Path]) -> str:
    captions = {
        "smartpnt_gt_trajectory_overview": "GT local X-Y trajectories for all available SmartPNT-MSF sequences after parsing and timestamp alignment.",
        "smartpnt_alignment_summary": "Median IMU magnitudes and GT speed after alignment; converted acceleration is gravity-scale after raw increment normalization.",
    }
    lines = ["# Figure Index", ""]
    stems = sorted({path.stem for path in figure_paths})
    for stem in stems:
        caption = captions.get(
            stem,
            "Short aligned IMU and GT speed sanity plot for one SmartPNT-MSF sequence.",
        )
        lines.append(f"- `{stem}`: {caption}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/smartpnt_msf"))
    parser.add_argument("--report-root", type=Path, default=Path("outputs/smartpnt_msf"))
    args = parser.parse_args()

    aligned_root = args.processed_root / "imu_gt_aligned"
    figure_root = args.report_root / "figures"
    aligned_paths = sorted(aligned_root.glob("*_imu_gt_aligned.parquet"))
    print(f"READS: {aligned_root} ({len(aligned_paths)} aligned parquet files)")
    print(f"WRITES: {args.report_root / 'alignment_quality.json'}, {args.report_root / 'alignment_quality.md'}, {figure_root}")
    print("SKIPS: raw ROS bag files; raw groundtruth text files")
    if not aligned_paths:
        raise FileNotFoundError(f"No aligned parquet files found under {aligned_root}")

    summaries = []
    frames = {}
    for path in aligned_paths:
        item, frame = _summarize_frame(path)
        summaries.append(item)
        frames[str(item["sequence_id"])] = frame

    figure_paths: list[Path] = []
    figure_paths.extend(_plot_trajectories(frames, figure_root))
    figure_paths.extend(_plot_alignment_summary(summaries, figure_root))
    first_key = sorted(frames)[0]
    figure_paths.extend(_plot_signal_sanity(frames[first_key], figure_root))

    summary = {
        "processed_root": str(args.processed_root),
        "figure_root": str(figure_root),
        "sequences": summaries,
        "duplicate_sequence_groups": _find_duplicate_content(summaries),
        "figures": [str(path) for path in figure_paths],
    }
    args.report_root.mkdir(parents=True, exist_ok=True)
    (args.report_root / "alignment_quality.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.report_root / "alignment_quality.md").write_text(_markdown_report(summary), encoding="utf-8")
    (figure_root / "figure_index.md").write_text(_figure_index(figure_paths), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
