from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.io.multi_imu_calibration import acquire_calibration_assets
from virtual_imu.io.multi_imu_prepare import prepare_multi_dataset
from virtual_imu.logging import setup_logging

app = typer.Typer(add_completion=False, help="Prepare calibrated strict low-to-high IMU pairs.")


@app.command()
def main(
    config: Path = typer.Option(
        Path("configs/multi_dataset_strict_prepare.yaml"),
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Strict multi-dataset preparation YAML.",
    ),
    dataset: list[str] = typer.Option(
        [],
        "--dataset",
        help="Dataset ID to include; repeat for multiple datasets.",
    ),
    fetch_calibration: bool = typer.Option(
        False,
        "--fetch-calibration",
        help="Acquire declared calibration assets before preparation.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Report reads and outputs without writing prepared data.",
    ),
    max_duration_s: Optional[float] = typer.Option(  # noqa: UP045
        None,
        "--max-duration-s",
        min=0.001,
        help="Bound source extraction and use an isolated output variant.",
    ),
) -> None:
    setup_logging()
    logger.info("READS strict preparation config {}", config)
    loaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise typer.BadParameter("Preparation config must contain a YAML mapping", param_hint="--config")
    loaded["_config_path"] = str(config.resolve())
    if max_duration_s is not None:
        variant = _bounded_variant(max_duration_s)
        loaded["max_duration_s"] = float(max_duration_s)
        loaded["output_variant"] = variant
        for key, default in (
            ("strict_output_root", "data/processed/multi_imu_strict"),
            ("diagnostic_output_root", "data/processed/multi_imu_diagnostics"),
        ):
            loaded[key] = str(Path(str(loaded.get(key, default))) / variant)
        logger.info("WRITES bounded outputs under variant {}", variant)

    if fetch_calibration:
        if dry_run:
            logger.info("SKIPS calibration acquisition because --dry-run is active")
        else:
            logger.info("WRITES declared calibration assets and calibration manifests")
            calibration_summary = acquire_calibration_assets(loaded)
            blocked = [
                record
                for record in calibration_summary["artifacts"]
                if not bool(record.get("training_eligible", False))
            ]
            for record in blocked:
                logger.warning(
                    "BLOCKS strict calibration asset {}: {}",
                    record.get("artifact_id"),
                    record.get("reason"),
                )

    logger.info("SKIPS datasets not selected by repeatable --dataset filters")
    summary = prepare_multi_dataset(loaded, datasets=dataset or None, dry_run=dry_run)
    if dry_run:
        output = str(loaded["strict_output_root"])
        typer.echo(f"DRY RUN strict output root: {output}")
    else:
        typer.echo(str(summary["strict_manifest"]))


def _bounded_variant(max_duration_s: float) -> str:
    value = f"{float(max_duration_s):g}".replace(".", "p")
    return f"bounded_{value}s"


if __name__ == "__main__":
    app()
