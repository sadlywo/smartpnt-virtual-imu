from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.training.penn_loso_runner import run_penn_loso_comparison_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the strict six-method Penn default LOSO comparison."
    )
    parser.add_argument("--config", type=Path, required=True, help="Comparison YAML path.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Train each neural method for one epoch with a small CPU network.",
    )
    args = parser.parse_args()
    print(f"READS kind=comparison_config path={args.config}")
    print("WRITES kind=comparison_run value=configured runs directory plus latest report snapshot")
    print("SKIPS kind=source_data_mutation value=all manifest and Parquet inputs remain read-only")
    run_dir = run_penn_loso_comparison_from_config(args.config, smoke=args.smoke)
    print(run_dir)


if __name__ == "__main__":
    main()
