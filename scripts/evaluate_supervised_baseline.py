from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.eval.supervised_baseline import (
    evaluate_supervised_baselines,
    write_supervised_baseline_outputs,
)
from virtual_imu.logging import setup_logging


def main(
    train: Path = typer.Option(..., "--train"),
    test: Path = typer.Option(..., "--test"),
    out: Path = typer.Option(..., "--out"),
    fir_taps: int = typer.Option(9, "--fir-taps"),
    ridge_alpha: float = typer.Option(1e-2, "--ridge-alpha"),
) -> None:
    setup_logging()
    logger.info("READS train split {}", train)
    logger.info("READS test split {}", test)
    logger.info("WRITES supervised baseline summaries rooted at {}", out)
    logger.info("SKIPS neural baselines, navigation metrics, Allan metrics, and prediction Parquet files")
    train_df = pd.read_parquet(train)
    test_df = pd.read_parquet(test)
    result = evaluate_supervised_baselines(
        train_df,
        test_df,
        fir_taps=fir_taps,
        ridge_alpha=ridge_alpha,
    )
    paths = write_supervised_baseline_outputs(result, out)
    typer.echo(paths["markdown"])


if __name__ == "__main__":
    typer.run(main)
