"""Run the scenario-aware minimal-component FIR/ARX experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from virtual_imu.fir_arx_research.minimal_pipeline import run_minimal_experiment  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/fir_arx/minimal.yaml")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(f"RUN_DIR={run_minimal_experiment(args.config, run_id=args.run_id)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
