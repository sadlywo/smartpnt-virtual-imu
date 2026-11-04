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


def _parse_prediction_specs(specs: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise typer.BadParameter(f"Prediction spec must look like NAME=path.parquet, got {spec!r}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise typer.BadParameter(f"Prediction spec has an empty method name: {spec!r}")
        parsed[name] = Path(path.strip())
    if not parsed:
        raise typer.BadParameter("At least one --prediction-spec NAME=path.parquet is required")
    return parsed


def main(
    supervised_json: Path = typer.Option(..., "--supervised-json"),
    prediction_spec: list[str] = typer.Option(..., "--prediction-spec"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    setup_logging()
    prediction_paths = _parse_prediction_specs(prediction_spec)
    logger.info("READS supervised baseline summary {}", supervised_json)
    for name, path in prediction_paths.items():
        logger.info("READS neural prediction table {} from {}", name, path)
    logger.info("WRITES comparison table rooted at {}", out)
    logger.info("SKIPS raw CSVs and checkpoint payloads")
    supervised = json.loads(supervised_json.read_text(encoding="utf-8"))
    predictions = {name: pd.read_parquet(path) for name, path in prediction_paths.items()}
    table = build_comparison_table(supervised, predictions)
    paths = write_comparison_table(table, out)
    typer.echo(paths["markdown"])


if __name__ == "__main__":
    typer.run(main)
