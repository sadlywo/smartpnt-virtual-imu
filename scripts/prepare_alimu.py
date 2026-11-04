from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.config import load_config
from virtual_imu.io.alimu_reader import prepare_alimu_dataset
from virtual_imu.logging import setup_logging


def prepare_alimu(config: str | Path) -> Path:
    setup_logging()
    cfg_path = Path(config)
    logger.info("READS config {}", cfg_path)
    cfg = load_config(cfg_path)
    logger.info("READS raw ALiMU CSVs under {}", cfg.get("raw_root", "data/raw/alimu/Low_Grade_High_Grade_IMU_dataset"))
    logger.info("WRITES processed Parquet and manifest under {}", cfg.get("processed_root", "data/processed/alimu"))
    logger.info("WRITES report to {}", cfg.get("report_path", "outputs/alimu/preparation_report.md"))
    include = cfg.get("include", {})
    skipped = []
    if not include.get("synthetic", False):
        skipped.append("Synthetic CSVs")
    if not include.get("real", True):
        skipped.append("Real CSVs")
    logger.info("SKIPS {}", ", ".join(skipped) if skipped else "none")
    result = prepare_alimu_dataset(cfg)
    return result["train_path"]


def main(config: Path = typer.Option(..., "--config")) -> None:
    train_path = prepare_alimu(config)
    typer.echo(train_path)


if __name__ == "__main__":
    typer.run(main)
