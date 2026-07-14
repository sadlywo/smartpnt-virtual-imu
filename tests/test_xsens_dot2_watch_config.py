from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad" / "freecad"))

from xsens_dot2_watch.config import WatchParameters, load_parameters  # noqa: E402


def test_default_parameters_cover_confirmed_wrist_range() -> None:
    params = load_parameters(ROOT / "configs" / "cad" / "xsens_dot2_watch.json")
    assert params.cavity_length_mm == pytest.approx(36.8)
    assert params.cavity_width_mm == pytest.approx(30.9)
    assert params.generated_wrist_min_mm == pytest.approx(140.0)
    assert params.generated_wrist_max_mm == pytest.approx(212.0)
    assert params.ratchet_positions == 19
    assert params.generated_wrist_max_mm >= params.wrist_target_max_mm


def test_invalid_clearance_is_rejected() -> None:
    params = WatchParameters(xy_clearance_per_side_mm=-0.1)
    with pytest.raises(ValueError, match="xy_clearance_per_side_mm"):
        params.validate()


def test_unknown_config_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"sensor_length_mm": 36.3, "mystery": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown parameter keys: mystery"):
        load_parameters(path)
