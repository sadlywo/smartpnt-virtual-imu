"""Build the configured unified labelled corpus without launching training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.io.unified_dataset import prepare_unified_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(f"READS: {args.config} and configured processed sources")
    print("WRITES: immutable processed dataset, splits, labels and source-of-truth build run")
    print("SKIPS: original raw datasets, duplicate training records, training launch")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = prepare_unified_dataset(config, config_path=args.config.resolve())
    print(f"BUILD_RUN: {result['build_run']}")
    print(f"COUNTS: {result['counts']}")


if __name__ == "__main__":
    main()
