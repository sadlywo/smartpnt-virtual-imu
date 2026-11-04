"""Run frozen paired and separate SmartPNT navigation experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.config import load_config
from virtual_imu.training.formal_pipeline import run_formal_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One epoch and first seed, stored separately; no hyperparameter tuning",
    )
    args = parser.parse_args()
    print(
        f"READS: {args.config}, frozen v2 manifests, configured source YAMLs, train/validation/test processed records",
        flush=True,
    )
    print(
        "WRITES: runs/formal_<training|smoke>_<time>/ configs, linked training runs, evaluations, aggregate metrics and styled figures",
        flush=True,
    )
    print(
        "SKIPS: raw/old/duplicate data, temperature/static branches, test-fitted calibration, candidate changes based on test results",
        flush=True,
    )
    run_formal_pipeline(load_config(args.config), config_path=args.config, smoke=args.smoke)


if __name__ == "__main__":
    main()
