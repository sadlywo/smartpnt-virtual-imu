from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.config import load_config
from virtual_imu.io.parquet_store import write_parquet
from virtual_imu.io.smartpnt_reader import SmartPNTReader
from virtual_imu.logging import setup_logging
from virtual_imu.preprocessing.split import split_by_fraction
from virtual_imu.preprocessing.sync import align_low_high
from virtual_imu.preprocessing.windowing import window_long_dataframe


def preparation_missing_report(cfg: dict[str, Any], inspection: dict[str, Any]) -> str:
    return (
        "# SmartPNT Preparation Report\n\n"
        "SmartPNT-MSF data is not ready for parsing yet.\n\n"
        f"- Raw root: `{cfg.get('raw_root', 'data/raw/smartpnt_msf/01_extracted')}`\n"
        f"- Inspection status: `{inspection.get('message', 'sources not configured')}`\n\n"
        "Next steps:\n\n"
        "1. Download archives under `data/raw/smartpnt_msf/00_archives` and extract usable folders under `data/raw/smartpnt_msf/01_extracted`.\n"
        "2. Run `python scripts/inspect_dataset.py --root data/raw/smartpnt_msf/01_extracted --out outputs/smartpnt_msf/inspection.md`.\n"
        "3. Set `low_imu_source`, `high_imu_source`, `ground_truth_source`, and column names in `configs/smartpnt_example.yaml`.\n"
    )


def preparation_report(aligned_path: Path, align_report: dict[str, Any], inspection: dict[str, Any]) -> str:
    lines = [
        "# SmartPNT Preparation Report",
        "",
        f"- Aligned output: `{aligned_path}`",
        f"- File count inspected: {inspection.get('file_count')}",
        "",
        "## Alignment",
        "",
    ]
    for key, value in align_report.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def prepare_smartpnt(config: str | Path) -> Path | None:
    setup_logging()
    cfg = load_config(config)
    out_dir = Path(cfg.get("output_dir", "data/processed/smartpnt_msf"))
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(cfg.get("report_path", "outputs/smartpnt_msf/preparation_report.md"))
    report_path.parent.mkdir(parents=True, exist_ok=True)

    reader = SmartPNTReader(cfg)
    inspection = reader.inspect()
    if not inspection.get("exists", False) or not cfg.get("low_imu_source") or not cfg.get("high_imu_source"):
        report_path.write_text(preparation_missing_report(cfg, inspection), encoding="utf-8")
        logger.warning("SmartPNT sources are not ready. Wrote guidance to {}", report_path)
        return None

    low, high = reader.read_pair()
    sync_cfg = cfg.get("sync", {})
    aligned, align_report = align_low_high(
        low,
        high,
        target_rate_hz=float(cfg.get("target_rate_hz", 100.0)),
        method=str(sync_cfg.get("method", "linear")),
        time_offset_s=float(cfg.get("time_offset_s", 0.0)),
        estimate_delay=bool(sync_cfg.get("estimate_delay", True)),
        max_lag_s=float(sync_cfg.get("max_lag_s", 0.5)),
    )
    aligned_path = write_parquet(aligned, out_dir / "aligned.parquet")
    splits = split_by_fraction(aligned, **cfg.get("split", {}))
    win_cfg = cfg.get("windowing", {})
    rate = float(cfg.get("target_rate_hz", 100.0))
    window_size = max(1, int(round(float(win_cfg.get("input_window_s", 2.0)) * rate)))
    stride = max(1, int(round(float(win_cfg.get("stride_s", 0.5)) * rate)))
    for name, frame in splits.items():
        windows = window_long_dataframe(frame, window_size, stride, target=str(win_cfg.get("target", "full_sequence")))
        write_parquet(windows, out_dir / f"windows_{name}.parquet")
    report_path.write_text(preparation_report(aligned_path, align_report, inspection), encoding="utf-8")
    logger.info("Prepared SmartPNT data under {}", out_dir)
    return aligned_path


def main(config: Path = typer.Option(..., "--config")) -> None:
    result = prepare_smartpnt(config)
    if result is not None:
        typer.echo(result)


if __name__ == "__main__":
    typer.run(main)