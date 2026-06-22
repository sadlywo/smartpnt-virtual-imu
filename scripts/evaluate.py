from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.cli import evaluate


def main(
    pred: Path = typer.Option(..., "--pred"),
    out: Path = typer.Option(..., "--out"),
    min_allan_duration_s: float = typer.Option(20.0, "--min-allan-duration-s"),
) -> None:
    evaluate(pred, out, min_allan_duration_s=min_allan_duration_s)


if __name__ == "__main__":
    typer.run(main)
