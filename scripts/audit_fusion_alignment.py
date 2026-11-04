"""Run the isolated FusionPortable timestamp/rotation audit from YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from virtual_imu.preprocessing.fusion_alignment_audit import run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/fusion_alignment_audit.yaml"))
    args = parser.parse_args()
    summary = run_audit(args.config)
    logger.info(
        "WRITES audit summary at {}; SKIPS cross-trajectory calibration claims",
        summary["metrics_path"],
    )


if __name__ == "__main__":
    main()
