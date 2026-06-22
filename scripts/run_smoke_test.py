from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virtual_imu.cli import run_smoke_test

if __name__ == "__main__":
    run_smoke_test()
