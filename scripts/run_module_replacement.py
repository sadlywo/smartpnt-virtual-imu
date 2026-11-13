"""Run the frozen CNN-LSTM front/back replacements on local CUDA."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")


if __name__ == "__main__":
    from virtual_imu.replacements.pipeline import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/ablations/cnn_lstm_module_replacement.yaml"
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--mode", choices=["all", "train", "evaluate", "preflight"], default="all")
    args = parser.parse_args()
    run(ROOT, args.config.resolve(), args.run_dir.resolve() if args.run_dir else None, args.mode)
