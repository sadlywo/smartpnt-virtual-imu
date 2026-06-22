from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.cli import prepare_smartpnt


def main(config: Path = typer.Option(..., "--config")) -> None:
    prepare_smartpnt(config)


if __name__ == "__main__":
    typer.run(main)
