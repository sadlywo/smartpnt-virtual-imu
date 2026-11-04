from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.config import load_config
from virtual_imu.io.alimu_context import prepare_context_splits


def main(config: Path = typer.Option(..., "--config")) -> None:
    cfg = load_config(config)
    typer.echo(f"READS: {config} and configured ALiMU split Parquets")
    manifest = prepare_context_splits(cfg)
    typer.echo(f"WRITES: {manifest['manifest_path']} and context split Parquets")
    typer.echo("SKIPS: raw CSVs, high-target-derived context, checkpoints, and run directories")
    typer.echo(manifest["manifest_path"])


if __name__ == "__main__":
    typer.run(main)
