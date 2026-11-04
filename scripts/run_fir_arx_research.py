"""Run a frozen FIR/ARX IMU-correction experiment.

This CLI reads the configured data manifest, assembles recording-disjoint
protocols, fits every selected model on TRAIN, selects hyperparameters on
VALIDATION, evaluates TEST once, and writes all artifacts under runs/.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from virtual_imu.fir_arx_research.pipeline import run_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/fir_arx/base.yaml",
        help="Experiment YAML",
    )
    parser.add_argument("--run-id", help="Optional deterministic run directory name")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_dir = run_experiment(args.config, run_id=args.run_id)
    print(f"RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
