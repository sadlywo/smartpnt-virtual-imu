from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.eval.comparison_table import build_comparison_table, write_comparison_table
from virtual_imu.logging import setup_logging


def main(
    supervised_json: Path = typer.Option(..., "--supervised-json"),
    prediction: Path = typer.Option(..., "--prediction"),
    out: Path = typer.Option(..., "--out"),
    neural_name: str = typer.Option("TCN", "--neural-name"),
) -> None:
    setup_logging()
    logger.info("READS supervised baseline summary {}", supervised_json)
    logger.info("READS neural prediction table {}", prediction)
    logger.info("WRITES comparison table rooted at {}", out)
    logger.info("SKIPS raw CSVs and model checkpoint payload")
    supervised = json.loads(supervised_json.read_text(encoding="utf-8"))
    pred_df = pd.read_parquet(prediction)
    table = build_comparison_table(supervised, pred_df, neural_name=neural_name)
    paths = write_comparison_table(table, out)
    typer.echo(paths["markdown"])


if __name__ == "__main__":
    typer.run(main)
