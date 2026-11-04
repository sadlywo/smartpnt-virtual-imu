from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.config import load_config
from virtual_imu.training.gt_trainer import train_gt_supervised_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(f"READS: {args.config}")
    print("WRITES: runs/<experiment_name_timestamp> with config, checkpoint, metrics, summary, test metrics")
    print("SKIPS: raw datasets; trainer reads processed aligned parquet files from config")
    run_dir = train_gt_supervised_model(load_config(args.config), config_path=args.config)
    print(run_dir)


if __name__ == "__main__":
    main()
