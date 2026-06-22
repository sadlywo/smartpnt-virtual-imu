from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.cli import inspect_dataset


def main(
    root: Path = typer.Option(Path("data/raw/smartpnt"), "--root"),
    out: Path | None = typer.Option(None, "--out"),
    preview_limit: int = typer.Option(5, "--preview-limit"),
) -> None:
    inspect_dataset(root, out=out, preview_limit=preview_limit)


if __name__ == "__main__":
    typer.run(main)
