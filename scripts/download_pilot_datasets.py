from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.io.pilot_download import run_pilot_download
from virtual_imu.logging import setup_logging


def main(
    config: Path = typer.Option(
        Path("configs/multi_dataset_pilot_download.yaml"),
        "--config",
        help="Pilot artifact catalog.",
    ),
    dataset: list[str] = typer.Option(
        [],
        "--dataset",
        help="Dataset name to include; repeat for multiple datasets.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Resolve sources without downloading."),
) -> None:
    setup_logging()
    logger.info("READS pilot catalog {}", config)
    logger.info("WRITES raw dataset roots and manifests declared by {}", config)
    logger.info("SKIPS unselected datasets, complete verified files, and non-IMU modalities")
    summary = run_pilot_download(config, datasets=dataset, dry_run=dry_run)
    for artifact in summary["artifacts"]:
        logger.info(
            "{} {} -> {} ({})",
            artifact["status"].upper(),
            artifact["artifact_id"],
            artifact["local_path"],
            artifact.get("extraction_status", "not_recorded"),
        )
        if artifact.get("reason"):
            logger.warning("BLOCKED {}: {}", artifact["artifact_id"], artifact["reason"])
    logger.info("WRITES pilot summary {}", summary["summary_path"])
    typer.echo(summary["summary_path"])


if __name__ == "__main__":
    typer.run(main)
