from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from virtual_imu.training.gt_supervised_dataset import (
    SMARTPNT_GT_IMU_COLUMNS,
    SmartPNTGTWindowDataset,
    make_smartpnt_gt_dataloader,
)


def _filter_sequences(df: pd.DataFrame, include: list[str], exclude: list[str]) -> pd.DataFrame:
    out = df.copy()
    if include:
        out = out[out["sequence_id"].astype(str).isin(include)]
    if exclude:
        out = out[~out["sequence_id"].astype(str).isin(exclude)]
    return out.reset_index(drop=True)


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = ["# SmartPNT-MSF GT-Supervised DataLoader Check", ""]
    lines.append(f"- Source root: `{summary['source_root']}`")
    lines.append(f"- Label mode: `{summary['label_mode']}`")
    lines.append(f"- Window size: `{summary['window_size']}`")
    lines.append(f"- Stride: `{summary['stride']}`")
    lines.append(f"- Batch size: `{summary['batch_size']}`")
    lines.append("")
    lines.append("## Batch")
    lines.append("")
    lines.append(f"- Input shape: `{summary['batch']['x_shape']}`")
    lines.append(f"- Target shape: `{summary['batch']['y_shape']}`")
    lines.append(f"- Input columns: `{summary['feature_columns']}`")
    lines.append(f"- Target columns: `{summary['target_columns']}`")
    lines.append("")
    lines.append("## Sequences")
    lines.append("")
    lines.append("| sequence | rows | windows |")
    lines.append("|---|---:|---:|")
    for item in summary["sequences"]:
        lines.append(f"| {item['sequence_id']} | {item['rows']} | {item['windows']} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Inputs are aligned IMU rate channels: `gx_radps/gy_radps/gz_radps/ax_mps2/ay_mps2/az_mps2`.")
    lines.append("- `velocity`, `position`, and `velocity_position` return sequence labels with shape `[channels, window]`.")
    lines.append("- `position_delta` and `pose_delta` return one short-window target vector per window.")
    lines.append("- Do not split duplicated `Traj1`/`Traj2` across train and test as independent generalization evidence.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/smartpnt_msf/imu_gt_aligned"))
    parser.add_argument("--label-mode", default="pose_delta")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--include-sequence", action="append", default=[])
    parser.add_argument("--exclude-sequence", action="append", default=[])
    parser.add_argument("--out", type=Path, default=Path("outputs/smartpnt_msf/gt_dataloader_check.json"))
    args = parser.parse_args()

    paths = sorted(args.data_root.glob("*_imu_gt_aligned.parquet")) if args.data_root.is_dir() else [args.data_root]
    print(f"READS: {args.data_root} ({len(paths)} aligned parquet files)")
    print(f"WRITES: {args.out}, {args.out.with_suffix('.md')}")
    print("SKIPS: raw ROS bags, raw GT text files, camera bags, GNSS files")
    if not paths:
        raise FileNotFoundError(f"No aligned SmartPNT GT parquet files found under {args.data_root}")

    df = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    df = _filter_sequences(df, args.include_sequence, args.exclude_sequence)
    dataset = SmartPNTGTWindowDataset(
        df,
        window_size=args.window_size,
        stride=args.stride,
        label_mode=args.label_mode,
        normalize_features=True,
        normalize_targets=True,
    )
    loader = make_smartpnt_gt_dataloader(
        df,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        label_mode=args.label_mode,
        feature_stats=dataset.feature_stats,
        target_stats=dataset.target_stats,
        normalize_features=True,
        normalize_targets=True,
        shuffle=False,
    )
    x, y, meta = next(iter(loader))
    sequence_rows = df.groupby("sequence_id", sort=False).size().to_dict()
    sequence_windows = {}
    for item in dataset.windows:
        sequence_windows[item["sequence_id"]] = sequence_windows.get(item["sequence_id"], 0) + 1
    summary = {
        "source_root": str(args.data_root),
        "label_mode": args.label_mode,
        "window_size": args.window_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "feature_columns": SMARTPNT_GT_IMU_COLUMNS,
        "target_columns": dataset.target_columns,
        "target_layout": dataset.target_layout,
        "feature_stats": dataset.feature_stats,
        "target_stats": dataset.target_stats,
        "sequences": [
            {
                "sequence_id": str(sequence_id),
                "rows": int(rows),
                "windows": int(sequence_windows.get(str(sequence_id), 0)),
            }
            for sequence_id, rows in sequence_rows.items()
        ],
        "batch": {
            "x_shape": list(x.shape),
            "y_shape": list(y.shape),
            "sequence_id": meta["sequence_id"],
            "start_index": [int(value) for value in meta["start_index"]],
            "end_index": [int(value) for value in meta["end_index"]],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out.with_suffix(".md").write_text(_markdown_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
