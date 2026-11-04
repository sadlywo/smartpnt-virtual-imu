from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.config import load_config
from virtual_imu.io.alimu_splits import write_strict_split
from virtual_imu.logging import setup_logging


def _config_value(config: dict[str, Any], cli_value: Any, key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return config.get(key, default)


def main(
    config: Path = typer.Option(None, "--config"),
    input_path: Path = typer.Option(None, "--input"),
    out_dir: Path = typer.Option(None, "--out-dir"),
    mode: str = typer.Option(None, "--mode"),
    holdout: str = typer.Option(None, "--holdout"),
    val_fraction: float = typer.Option(None, "--val-fraction"),
    seed: int = typer.Option(None, "--seed"),
) -> None:
    setup_logging()
    cfg = load_config(config) if config else {}
    source = Path(_config_value(cfg, input_path, "input_path", "data/processed/alimu/all.parquet"))
    output_root = Path(_config_value(cfg, out_dir, "out_dir", "data/processed/alimu_strict/default"))
    split_mode = str(_config_value(cfg, mode, "mode", "leave_sensor_out"))
    split_holdout = str(_config_value(cfg, holdout, "holdout", "MPU9250"))
    split_val_fraction = float(_config_value(cfg, val_fraction, "val_fraction", 0.2))
    split_seed = int(_config_value(cfg, seed, "seed", 7))

    logger.info("READS config {}", config if config else "none")
    logger.info("READS processed ALiMU table {}", source)
    logger.info("WRITES strict train/val/test Parquet and manifest under {}", output_root)
    logger.info("SKIPS raw CSVs, model checkpoints, and generated run directories")
    df = pd.read_parquet(source)
    result = write_strict_split(
        df,
        source_path=source,
        output_root=output_root,
        mode=split_mode,
        holdout=split_holdout,
        val_fraction=split_val_fraction,
        seed=split_seed,
    )
    typer.echo(result["manifest_path"])


if __name__ == "__main__":
    typer.run(main)
