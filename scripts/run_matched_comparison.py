"""Run the complete matched ALiMU baseline and diffusion ablation experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_imu.diffusion.matched import run_matched_experiment  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ddpm/alimu_matched.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    path = run_matched_experiment(args.config, ROOT, args.run_id, args.resume)
    print(f"RUN_DIR={path}", flush=True)


if __name__ == "__main__":
    main()
