"""Generate metadata-only, recording-disjoint reviewer experiment protocols."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from virtual_imu.io.reviewer_protocol_splits import generate_reviewer_protocol_splits
from virtual_imu.utils.paths import project_root_from_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reviewer_splits.yaml"))
    arguments = parser.parse_args()
    print(f"READS: reviewer split config {arguments.config}", flush=True)
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    result = generate_reviewer_protocol_splits(config, project_root_from_file(__file__))
    print(f"WRITES: protocol index {result['index_path']} protocols={len(result['protocols'])}", flush=True)


if __name__ == "__main__":
    main()
