"""Run the complete controlled shared CNN-BiLSTM ablation on local CUDA."""
from pathlib import Path
import argparse
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from virtual_imu.ablations.pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ablations/cnn_bilstm_shared.yaml")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--mode", choices=["all", "train", "evaluate", "preflight"], default="all")
    args = parser.parse_args()
    run(ROOT, args.config.resolve(), args.run_dir.resolve() if args.run_dir else None, args.mode)
