from __future__ import annotations

import sys
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.cli import download_smartpnt
from virtual_imu.logging import setup_logging


def main(
    repo_url: str = typer.Option("https://gitee.com/lv-jiarui/SmartPNT-MSF-Datasets", "--repo-url"),
    out: Path = typer.Option(Path("data/raw/smartpnt"), "--out"),
    manifest: Path = typer.Option(Path("configs/smartpnt_manifest.yaml"), "--manifest"),
    sha256: bool = typer.Option(False, "--sha256"),
) -> None:
    setup_logging()
    download_smartpnt(repo_url, out, manifest_config=manifest, include_sha256=sha256)


if __name__ == "__main__":
    typer.run(main)
