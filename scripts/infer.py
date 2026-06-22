from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.cli import infer


def main(
    checkpoint: Path = typer.Option(..., "--checkpoint"),
    input_path: Path = typer.Option(..., "--input"),
    out: Path = typer.Option(..., "--out"),
    device: str = typer.Option("auto", "--device"),
) -> None:
    infer(checkpoint, input_path, out, device=device)


if __name__ == "__main__":
    typer.run(main)
