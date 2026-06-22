from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.cli import plot_results


def main(
    eval_path: Path = typer.Option(..., "--eval"),
    pred: Path = typer.Option(..., "--pred"),
    out: Path = typer.Option(..., "--out"),
) -> None:
    plot_results(eval_path, pred, out)


if __name__ == "__main__":
    typer.run(main)
