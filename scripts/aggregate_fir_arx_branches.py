"""Compare VMD, TDA, and minimal-component FIR/ARX branches to raw controls."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from virtual_imu.fir_arx_research.aggregate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
