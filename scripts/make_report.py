from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.cli import make_report


def main(
    run: Path = typer.Option(..., "--run"),
    out: Path = typer.Option(..., "--out"),
    metrics: Path | None = typer.Option(None, "--metrics"),
    figures: Path | None = typer.Option(None, "--figures"),
    baselines: Path | None = typer.Option(None, "--baselines"),
) -> None:
    make_report(run, out, metrics=metrics, figures=figures, baselines=baselines)


if __name__ == "__main__":
    typer.run(main)
