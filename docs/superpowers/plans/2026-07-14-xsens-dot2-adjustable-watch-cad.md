# Xsens DOT 2 Adjustable Watch CAD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a parameterized FreeCAD 0.21.1 generator for an FDM-printable Xsens DOT 2 wrist mount, then provide a SolidWorks 2022 VBA macro that imports the generated STEP components into a native assembly.

**Architecture:** Keep printer-independent parameters and artifact bookkeeping in pure Python, while isolating FreeCAD imports inside focused geometry modules. The FreeCAD macro creates one timestamped build containing FCStd, STEP, STL, parameter, manifest, and log artifacts; a separate SolidWorks VBA module consumes a CSV import manifest and never recreates the geometry.

**Tech Stack:** Python 3.8 inside FreeCAD 0.21.1, OpenCascade through FreeCAD `Part`, FreeCAD `Mesh`, host Python 3.11 with pytest/ruff, PowerShell, and SolidWorks 2022 VBA/API.

---

## Scope and Source References

Implement the approved design in `docs/superpowers/specs/2026-07-14-xsens-dot2-adjustable-watch-cad-design.md`. Use the official Xsens envelope of 36.3 × 30.4 × 10.8 mm, keep the top button and LED exposed, and require sensor removal for charging. Do not introduce metal or magnetic hardware.

The SolidWorks implementation must follow the 2022 API patterns for `ISldWorks.LoadFile4`, `IAssemblyDoc.AddComponent5`, `IModelDoc2.AddConfiguration3`, `IAssemblyDoc.SetComponentState`, and `IModelDocExtension.SaveAs`.

Primary references:

- [Xsens DOT User Manual](https://www.movella.com/hubfs/Xsens%20DOT%20User%20Manual.pdf)
- [FreeCAD Python scripting basics](https://reqrefusion.github.io/FreeCAD-Documentation-html/wiki/FreeCAD_Scripting_Basics.html)
- [FreeCAD Part API](https://reqrefusion.github.io/FreeCAD-Documentation-html/wiki/Part_API.html)
- [SolidWorks 2022 Import STEP File example](https://help.solidworks.com/2022/english/api/sldworksapi/import_step_file_example_vb.htm)
- [SolidWorks 2022 Add Component and Mate example](https://help.solidworks.com/2022/English/api/sldworksapi/add_component_and_mate_example_vb.htm)
- [SolidWorks 2022 `IModelDoc2.IAddConfiguration3`](https://help.solidworks.com/2022/english/api/sldworksapi/SOLIDWORKS.Interop.sldworks~SOLIDWORKS.Interop.sldworks.IModelDoc2~IAddConfiguration3.html)
- [SolidWorks 2022 `IAssemblyDoc.SetComponentState`](https://help.solidworks.com/2022/english/api/sldworksapi/SolidWorks.Interop.sldworks~SolidWorks.Interop.sldworks.IAssemblyDoc~SetComponentState.html)

## File Map

| Path | Responsibility |
|---|---|
| `configs/cad/xsens_dot2_watch.json` | Versioned default dimensions, clearances, materials, and output settings |
| `cad/freecad/Build_Xsens_DOT2_Watch.FCMacro` | Thin GUI/headless FreeCAD entry point |
| `cad/freecad/xsens_dot2_watch/config.py` | Python 3.8-compatible parameter dataclass and validation |
| `cad/freecad/xsens_dot2_watch/artifacts.py` | Timestamped directories, hashes, JSON, and build logging |
| `cad/freecad/xsens_dot2_watch/geometry.py` | Shared OpenCascade solid helpers |
| `cad/freecad/xsens_dot2_watch/sensor.py` | DOT 2 fit dummy |
| `cad/freecad/xsens_dot2_watch/cradle.py` | PETG cradle and fit coupon |
| `cad/freecad/xsens_dot2_watch/straps.py` | TPU long/short straps and ratchet teeth |
| `cad/freecad/xsens_dot2_watch/ratchet.py` | PETG housing, pawl, and polymer pin |
| `cad/freecad/xsens_dot2_watch/assembly.py` | Component metadata, placements, and assembly document |
| `cad/freecad/xsens_dot2_watch/validation.py` | Shape, interference, STEP, STL, and range checks |
| `cad/freecad/xsens_dot2_watch/exporter.py` | FCStd/STEP/STL export and import CSV generation |
| `cad/freecad/xsens_dot2_watch/build.py` | End-to-end orchestration |
| `cad/freecad/tests/run_freecad_geometry_tests.py` | FreeCADCmd-native geometry regression tests |
| `cad/freecad/tests/verify_build.py` | Host-Python artifact and hash verifier |
| `scripts/build_xsens_dot2_watch.ps1` | Repeatable Windows CLI wrapper with read/write/skip logging |
| `cad/solidworks/Import_Xsens_DOT2_Watch.bas` | SolidWorks 2022 STEP import and assembly macro |
| `tests/test_xsens_dot2_watch_config.py` | Pure parameter/range tests |
| `tests/test_xsens_dot2_watch_artifacts.py` | Pure artifact/logging tests |
| `tests/test_xsens_dot2_solidworks_macro.py` | Static VBA contract checks where SolidWorks is unavailable |
| `docs/CAD_XSENS_DOT2_WATCH.md` | User build, print, fit-calibration, and SolidWorks instructions |

Do not commit anything below `outputs/cad/`.

### Task 1: Establish the parameter contract

**Files:**
- Create: `configs/cad/xsens_dot2_watch.json`
- Create: `cad/freecad/xsens_dot2_watch/__init__.py`
- Create: `cad/freecad/xsens_dot2_watch/config.py`
- Create: `tests/test_xsens_dot2_watch_config.py`

- [ ] **Step 1: Write the failing pure-Python parameter tests**

```python
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
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run:

```powershell
uv run pytest tests/test_xsens_dot2_watch_config.py -q
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'xsens_dot2_watch'`.

- [ ] **Step 3: Add the checked-in default JSON**

```json
{
  "sensor_length_mm": 36.3,
  "sensor_width_mm": 30.4,
  "sensor_height_mm": 10.8,
  "sensor_corner_radius_mm": 6.0,
  "xy_clearance_per_side_mm": 0.25,
  "cradle_length_mm": 44.8,
  "cradle_width_mm": 39.0,
  "cradle_corner_radius_mm": 8.0,
  "cradle_base_thickness_mm": 3.0,
  "cradle_wall_thickness_mm": 2.4,
  "cradle_low_wall_height_mm": 4.0,
  "snap_post_length_mm": 8.0,
  "snap_post_thickness_mm": 2.0,
  "snap_hook_height_mm": 1.2,
  "snap_engagement_mm": 0.6,
  "strap_width_mm": 20.0,
  "strap_thickness_mm": 2.2,
  "long_strap_length_mm": 120.0,
  "short_strap_length_mm": 40.0,
  "ratchet_pitch_mm": 4.0,
  "ratchet_tooth_height_mm": 1.2,
  "ratchet_positions": 19,
  "ratchet_housing_length_mm": 28.0,
  "ratchet_housing_width_mm": 24.0,
  "ratchet_housing_height_mm": 7.0,
  "ratchet_slide_clearance_per_side_mm": 0.35,
  "pin_diameter_mm": 1.75,
  "pin_hole_clearance_mm": 0.25,
  "wrist_target_min_mm": 140.0,
  "wrist_target_max_mm": 210.0,
  "petg_material": "PETG",
  "tpu_material": "TPU 95A",
  "stl_linear_deflection_mm": 0.08,
  "stl_angular_deflection_deg": 12.0
}
```

- [ ] **Step 4: Implement the Python 3.8-compatible parameter object**

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class WatchParameters:
    sensor_length_mm: float = 36.3
    sensor_width_mm: float = 30.4
    sensor_height_mm: float = 10.8
    sensor_corner_radius_mm: float = 6.0
    xy_clearance_per_side_mm: float = 0.25
    cradle_length_mm: float = 44.8
    cradle_width_mm: float = 39.0
    cradle_corner_radius_mm: float = 8.0
    cradle_base_thickness_mm: float = 3.0
    cradle_wall_thickness_mm: float = 2.4
    cradle_low_wall_height_mm: float = 4.0
    snap_post_length_mm: float = 8.0
    snap_post_thickness_mm: float = 2.0
    snap_hook_height_mm: float = 1.2
    snap_engagement_mm: float = 0.6
    strap_width_mm: float = 20.0
    strap_thickness_mm: float = 2.2
    long_strap_length_mm: float = 120.0
    short_strap_length_mm: float = 40.0
    ratchet_pitch_mm: float = 4.0
    ratchet_tooth_height_mm: float = 1.2
    ratchet_positions: int = 19
    ratchet_housing_length_mm: float = 28.0
    ratchet_housing_width_mm: float = 24.0
    ratchet_housing_height_mm: float = 7.0
    ratchet_slide_clearance_per_side_mm: float = 0.35
    pin_diameter_mm: float = 1.75
    pin_hole_clearance_mm: float = 0.25
    wrist_target_min_mm: float = 140.0
    wrist_target_max_mm: float = 210.0
    petg_material: str = "PETG"
    tpu_material: str = "TPU 95A"
    stl_linear_deflection_mm: float = 0.08
    stl_angular_deflection_deg: float = 12.0

    @property
    def cavity_length_mm(self) -> float:
        return self.sensor_length_mm + 2.0 * self.xy_clearance_per_side_mm

    @property
    def cavity_width_mm(self) -> float:
        return self.sensor_width_mm + 2.0 * self.xy_clearance_per_side_mm

    @property
    def fixed_loop_length_mm(self) -> float:
        return (
            self.cradle_length_mm
            + self.long_strap_length_mm
            + self.short_strap_length_mm
            + self.ratchet_housing_length_mm
        )

    @property
    def maximum_overlap_mm(self) -> float:
        return self.fixed_loop_length_mm - self.wrist_target_min_mm

    @property
    def minimum_overlap_mm(self) -> float:
        return self.maximum_overlap_mm - (self.ratchet_positions - 1) * self.ratchet_pitch_mm

    @property
    def generated_wrist_min_mm(self) -> float:
        return self.fixed_loop_length_mm - self.maximum_overlap_mm

    @property
    def generated_wrist_max_mm(self) -> float:
        return self.fixed_loop_length_mm - self.minimum_overlap_mm

    def validate(self) -> "WatchParameters":
        values = asdict(self)
        for name, value in values.items():
            if isinstance(value, (int, float)) and name != "ratchet_positions" and value <= 0:
                raise ValueError("%s must be > 0" % name)
        if self.ratchet_positions < 2:
            raise ValueError("ratchet_positions must be >= 2")
        if self.xy_clearance_per_side_mm <= 0:
            raise ValueError("xy_clearance_per_side_mm must be > 0")
        if self.wrist_target_min_mm >= self.wrist_target_max_mm:
            raise ValueError("wrist_target_min_mm must be smaller than wrist_target_max_mm")
        if self.minimum_overlap_mm <= 0:
            raise ValueError("ratchet field is longer than available overlap")
        if self.generated_wrist_max_mm < self.wrist_target_max_mm:
            raise ValueError("ratchet positions do not cover wrist_target_max_mm")
        return self

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "cavity_length_mm": self.cavity_length_mm,
                "cavity_width_mm": self.cavity_width_mm,
                "generated_wrist_min_mm": self.generated_wrist_min_mm,
                "generated_wrist_max_mm": self.generated_wrist_max_mm,
                "minimum_overlap_mm": self.minimum_overlap_mm,
                "maximum_overlap_mm": self.maximum_overlap_mm,
            }
        )
        return result


def load_parameters(path: Path) -> WatchParameters:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {item.name for item in fields(WatchParameters)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError("Unknown parameter keys: %s" % ", ".join(unknown))
    return WatchParameters(**raw).validate()
```

Set `cad/freecad/xsens_dot2_watch/__init__.py` to:

```python
from .config import WatchParameters, load_parameters

__all__ = ["WatchParameters", "load_parameters"]
```

- [ ] **Step 5: Run the focused and lint tests**

Run:

```powershell
uv run pytest tests/test_xsens_dot2_watch_config.py -q
uv run ruff check tests/test_xsens_dot2_watch_config.py
uv run ruff check --target-version py38 cad/freecad/xsens_dot2_watch
```

Expected: both commands exit 0; pytest reports 3 passed.

- [ ] **Step 6: Commit the parameter contract**

```powershell
git add configs/cad/xsens_dot2_watch.json cad/freecad/xsens_dot2_watch tests/test_xsens_dot2_watch_config.py
git commit -m "feat: define Xsens DOT 2 watch CAD parameters"
```

### Task 2: Add timestamped artifact and logging infrastructure

**Files:**
- Create: `cad/freecad/xsens_dot2_watch/artifacts.py`
- Create: `tests/test_xsens_dot2_watch_artifacts.py`

- [ ] **Step 1: Write failing tests for immutable run directories and log categories**

```python
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "cad" / "freecad"))

from xsens_dot2_watch.artifacts import BuildLog, create_run_directory, sha256_file, write_json  # noqa: E402


def test_create_run_directory_is_timestamped_and_non_overwriting(tmp_path: Path) -> None:
    now = datetime(2026, 7, 14, 21, 30, 45)
    first = create_run_directory(tmp_path, now)
    second = create_run_directory(tmp_path, now)
    assert first.name == "20260714T213045"
    assert second.name == "20260714T213045_01"


def test_build_log_records_read_write_skip(tmp_path: Path) -> None:
    path = tmp_path / "build_log.txt"
    log = BuildLog(path)
    log.read("config.json")
    log.write("part.step")
    log.skip("existing.stl", "already exists")
    text = path.read_text(encoding="utf-8")
    assert "READ config.json" in text
    assert "WRITE part.step" in text
    assert "SKIP existing.stl reason=already exists" in text


def test_json_and_hash_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "parameters.json"
    write_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert sha256_file(path) == sha256_file(path)
```

- [ ] **Step 2: Verify the tests fail because `artifacts.py` is missing**

Run `uv run pytest tests/test_xsens_dot2_watch_artifacts.py -q`.

Expected: FAIL with `ModuleNotFoundError` for `xsens_dot2_watch.artifacts`.

- [ ] **Step 3: Implement artifact helpers without FreeCAD imports**

```python
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class BuildLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def _line(self, category: str, target: str, detail: str = "") -> None:
        suffix = " " + detail if detail else ""
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("%s %s%s\n" % (category, target, suffix))

    def read(self, target: str) -> None:
        self._line("READ", target)

    def write(self, target: str) -> None:
        self._line("WRITE", target)

    def skip(self, target: str, reason: str) -> None:
        self._line("SKIP", target, "reason=%s" % reason)

    def fail(self, stage: str, reason: str) -> None:
        self._line("FAIL", stage, "reason=%s" % reason)


def create_run_directory(root: Path, now: datetime) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stem = now.strftime("%Y%m%dT%H%M%S")
    for index in range(100):
        name = stem if index == 0 else "%s_%02d" % (stem, index)
        candidate = root / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError("Unable to allocate timestamped build directory under %s" % root)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
```

- [ ] **Step 4: Run tests and lint**

Run:

```powershell
uv run pytest tests/test_xsens_dot2_watch_artifacts.py -q
uv run ruff check tests/test_xsens_dot2_watch_artifacts.py
uv run ruff check --target-version py38 cad/freecad/xsens_dot2_watch/artifacts.py
```

Expected: pytest reports 3 passed and ruff exits 0.

- [ ] **Step 5: Commit artifact infrastructure**

```powershell
git add cad/freecad/xsens_dot2_watch/artifacts.py tests/test_xsens_dot2_watch_artifacts.py
git commit -m "feat: add CAD artifact traceability"
```

### Task 3: Build shared FreeCAD primitives and the DOT 2 dummy

**Files:**
- Create: `cad/freecad/xsens_dot2_watch/geometry.py`
- Create: `cad/freecad/xsens_dot2_watch/sensor.py`
- Create: `cad/freecad/tests/run_freecad_geometry_tests.py`

- [ ] **Step 1: Add a FreeCAD-native failing test runner**

```python
from __future__ import print_function

from pathlib import Path

from xsens_dot2_watch.config import load_parameters
from xsens_dot2_watch.geometry import rounded_box
from xsens_dot2_watch.sensor import build_sensor_dummy

ROOT = Path(__file__).resolve().parents[3]
PARAMS = load_parameters(ROOT / "configs" / "cad" / "xsens_dot2_watch.json")


def assert_close(actual, expected, tolerance=1e-6):
    if abs(actual - expected) > tolerance:
        raise AssertionError("%r != %r" % (actual, expected))


def test_rounded_box():
    shape = rounded_box(20.0, 10.0, 3.0, 2.0)
    if not shape.isValid() or len(shape.Solids) != 1:
        raise AssertionError("rounded_box must return one valid solid")
    assert_close(shape.BoundBox.XLength, 20.0)
    assert_close(shape.BoundBox.YLength, 10.0)
    assert_close(shape.BoundBox.ZLength, 3.0)


def test_sensor_dummy():
    shape = build_sensor_dummy(PARAMS)
    if not shape.isValid() or len(shape.Solids) != 1:
        raise AssertionError("sensor dummy must return one valid solid")
    assert_close(shape.BoundBox.XLength, PARAMS.sensor_length_mm)
    assert_close(shape.BoundBox.YLength, PARAMS.sensor_width_mm)
    assert_close(shape.BoundBox.ZLength, PARAMS.sensor_height_mm)


test_rounded_box()
test_sensor_dummy()
print("FREECAD_TEST_PASS primitives sensor")
```

- [ ] **Step 2: Run FreeCADCmd and verify the missing-module failure**

Run:

```powershell
& 'E:\FreeCAD\bin\FreeCADCmd.exe' -P "$PWD\cad\freecad" "$PWD\cad\freecad\tests\run_freecad_geometry_tests.py"
```

Expected: non-zero exit or traceback showing `xsens_dot2_watch.geometry` is missing.

- [ ] **Step 3: Implement deterministic rounded solids**

```python
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import Part
from FreeCAD import Vector


def fuse_all(shapes: Iterable[Part.Shape]) -> Part.Shape:
    items = list(shapes)
    if not items:
        raise ValueError("fuse_all requires at least one shape")
    result = items[0]
    for shape in items[1:]:
        result = result.fuse(shape)
    return result.removeSplitter()


def rounded_box(length: float, width: float, height: float, radius: float) -> Part.Shape:
    if min(length, width, height, radius) <= 0:
        raise ValueError("rounded_box dimensions must be positive")
    if 2.0 * radius >= min(length, width):
        raise ValueError("rounded_box radius is too large")
    x0 = -length / 2.0
    y0 = -width / 2.0
    pieces = [
        Part.makeBox(length - 2.0 * radius, width, height, Vector(x0 + radius, y0, 0)),
        Part.makeBox(length, width - 2.0 * radius, height, Vector(x0, y0 + radius, 0)),
    ]
    for x in (x0 + radius, x0 + length - radius):
        for y in (y0 + radius, y0 + width - radius):
            pieces.append(Part.makeCylinder(radius, height, Vector(x, y, 0)))
    return fuse_all(pieces)


def require_single_solid(name: str, shape: Part.Shape) -> Part.Shape:
    if shape.isNull() or not shape.isValid() or len(shape.Solids) != 1 or shape.Volume <= 0:
        raise RuntimeError("%s is not one valid non-zero solid" % name)
    return shape.removeSplitter()
```

- [ ] **Step 4: Implement the sensor envelope**

```python
from __future__ import annotations

import Part

from .config import WatchParameters
from .geometry import require_single_solid, rounded_box


def build_sensor_dummy(params: WatchParameters) -> Part.Shape:
    shape = rounded_box(
        params.sensor_length_mm,
        params.sensor_width_mm,
        params.sensor_height_mm,
        params.sensor_corner_radius_mm,
    )
    return require_single_solid("DOT2_Fit_Dummy", shape)
```

- [ ] **Step 5: Run the FreeCAD test runner**

Run the FreeCADCmd command from Step 2.

Expected: exit 0 and output contains `FREECAD_TEST_PASS primitives sensor`.

- [ ] **Step 6: Commit primitives and dummy**

```powershell
git add cad/freecad/xsens_dot2_watch/geometry.py cad/freecad/xsens_dot2_watch/sensor.py cad/freecad/tests/run_freecad_geometry_tests.py
git commit -m "feat: add FreeCAD geometry primitives and DOT dummy"
```

### Task 4: Generate the PETG cradle and fit coupon

**Files:**
- Create: `cad/freecad/xsens_dot2_watch/cradle.py`
- Modify: `cad/freecad/tests/run_freecad_geometry_tests.py`

- [ ] **Step 1: Add failing cradle and coupon assertions**

Place the import beside the existing `xsens_dot2_watch` imports, place the function before the final invocation block, and add its call to that block.

```python
from xsens_dot2_watch.cradle import build_cradle, build_fit_coupon


def test_cradle_and_coupon():
    cradle = build_cradle(PARAMS)
    coupon = build_fit_coupon(PARAMS)
    for name, shape in (("cradle", cradle), ("coupon", coupon)):
        if not shape.isValid() or len(shape.Solids) != 1 or shape.Volume <= 0:
            raise AssertionError("%s must be one valid solid" % name)
    assert_close(cradle.BoundBox.XLength, 60.8)
    assert_close(cradle.BoundBox.YLength, PARAMS.cradle_width_mm)
    if coupon.Volume >= cradle.Volume:
        raise AssertionError("fit coupon must be smaller than cradle")


test_cradle_and_coupon()
```

The 60.8 mm X length is the 44.8 mm body plus two 8.0 mm lug extensions.

- [ ] **Step 2: Run FreeCADCmd and verify the missing cradle module**

Expected: traceback naming `xsens_dot2_watch.cradle`.

- [ ] **Step 3: Implement the cradle as fused base, low ring, posts, hooks, and drilled lugs**

```python
from __future__ import annotations

import Part
from FreeCAD import Vector

from .config import WatchParameters
from .geometry import fuse_all, require_single_solid, rounded_box


def _lug(params: WatchParameters, side: int) -> Part.Shape:
    body_half = params.cradle_length_mm / 2.0
    x0 = body_half if side > 0 else -body_half - 8.0
    block = Part.makeBox(8.0, 24.0, 5.0, Vector(x0, -12.0, 0.0))
    center_x = x0 + 4.0
    hole_radius = (params.pin_diameter_mm + params.pin_hole_clearance_mm) / 2.0
    hole = Part.makeCylinder(
        hole_radius,
        26.0,
        Vector(center_x, -13.0, 2.8),
        Vector(0.0, 1.0, 0.0),
    )
    return block.cut(hole)


def build_cradle(params: WatchParameters) -> Part.Shape:
    base = rounded_box(
        params.cradle_length_mm,
        params.cradle_width_mm,
        params.cradle_base_thickness_mm,
        params.cradle_corner_radius_mm,
    )
    outer_length = params.cavity_length_mm + 2.0 * params.cradle_wall_thickness_mm
    outer_width = params.cavity_width_mm + 2.0 * params.cradle_wall_thickness_mm
    outer_radius = params.sensor_corner_radius_mm + params.xy_clearance_per_side_mm + params.cradle_wall_thickness_mm
    outer = rounded_box(outer_length, outer_width, params.cradle_low_wall_height_mm, outer_radius)
    inner = rounded_box(
        params.cavity_length_mm,
        params.cavity_width_mm,
        params.cradle_low_wall_height_mm + 0.2,
        params.sensor_corner_radius_mm + params.xy_clearance_per_side_mm,
    )
    ring = outer.cut(inner)
    ring.translate(Vector(0.0, 0.0, params.cradle_base_thickness_mm))

    post_z = params.cradle_base_thickness_mm
    top_y = params.cavity_width_mm / 2.0
    bottom_y = -top_y - params.snap_post_thickness_mm
    post_x = -params.snap_post_length_mm / 2.0
    post_height = params.sensor_height_mm
    top_post = Part.makeBox(
        params.snap_post_length_mm,
        params.snap_post_thickness_mm,
        post_height,
        Vector(post_x, top_y, post_z),
    )
    bottom_post = Part.makeBox(
        params.snap_post_length_mm,
        params.snap_post_thickness_mm,
        post_height,
        Vector(post_x, bottom_y, post_z),
    )
    hook_z = post_z + params.sensor_height_mm
    top_hook = Part.makeBox(
        params.snap_post_length_mm,
        params.snap_post_thickness_mm + params.snap_engagement_mm,
        params.snap_hook_height_mm,
        Vector(post_x, top_y - params.snap_engagement_mm, hook_z),
    )
    bottom_hook = Part.makeBox(
        params.snap_post_length_mm,
        params.snap_post_thickness_mm + params.snap_engagement_mm,
        params.snap_hook_height_mm,
        Vector(post_x, bottom_y, hook_z),
    )
    shape = fuse_all(
        [base, ring, top_post, bottom_post, top_hook, bottom_hook, _lug(params, -1), _lug(params, 1)]
    )
    return require_single_solid("DOT2_Cradle_PETG", shape)


def build_fit_coupon(params: WatchParameters) -> Part.Shape:
    cradle = build_cradle(params)
    cutter = Part.makeBox(20.0, params.cradle_width_mm / 2.0, 20.0, Vector(-10.0, 0.0, 0.0))
    return require_single_solid("DOT2_Fit_Coupon_PETG", cradle.common(cutter))
```

- [ ] **Step 4: Run FreeCADCmd tests and inspect exact bounds**

Run the FreeCADCmd test command. Expected: exit 0 and the existing pass marker. If OpenCascade returns a numeric tolerance difference, keep the assertion tolerance at `1e-6`; do not round the geometry to make the test pass.

- [ ] **Step 5: Commit cradle geometry**

```powershell
git add cad/freecad/xsens_dot2_watch/cradle.py cad/freecad/tests/run_freecad_geometry_tests.py
git commit -m "feat: generate PETG DOT cradle and fit coupon"
```

### Task 5: Generate the two TPU straps and discrete wrist positions

**Files:**
- Create: `cad/freecad/xsens_dot2_watch/straps.py`
- Modify: `cad/freecad/tests/run_freecad_geometry_tests.py`

- [ ] **Step 1: Add failing strap and range assertions**

Place the import beside the existing package imports and the function before the final invocation block.

```python
from xsens_dot2_watch.straps import build_long_strap, build_short_strap, wrist_positions_mm


def test_straps_and_positions():
    long_strap = build_long_strap(PARAMS)
    short_strap = build_short_strap(PARAMS)
    for name, shape in (("long_strap", long_strap), ("short_strap", short_strap)):
        if not shape.isValid() or len(shape.Solids) != 1 or shape.Volume <= 0:
            raise AssertionError("%s must be one valid solid" % name)
    positions = wrist_positions_mm(PARAMS)
    if len(positions) != 19:
        raise AssertionError("expected 19 ratchet positions")
    assert_close(positions[0], 140.0)
    assert_close(positions[-1], 212.0)
    for left, right in zip(positions, positions[1:]):
        assert_close(right - left, 4.0)


test_straps_and_positions()
```

- [ ] **Step 2: Run FreeCADCmd and verify failure on the missing module**

Expected: traceback naming `xsens_dot2_watch.straps`.

- [ ] **Step 3: Implement the straps and 19 fused ramp teeth**

```python
from __future__ import annotations

from typing import List

import Part
from FreeCAD import Vector

from .config import WatchParameters
from .geometry import fuse_all, require_single_solid


def _hinge_tab(params: WatchParameters, center_x: float) -> Part.Shape:
    tab = Part.makeBox(8.0, params.strap_width_mm, 5.0, Vector(center_x - 4.0, -params.strap_width_mm / 2.0, 0.0))
    radius = (params.pin_diameter_mm + params.pin_hole_clearance_mm) / 2.0
    hole = Part.makeCylinder(
        radius,
        params.strap_width_mm + 2.0,
        Vector(center_x, -params.strap_width_mm / 2.0 - 1.0, 2.8),
        Vector(0.0, 1.0, 0.0),
    )
    return tab.cut(hole)


def _ramp_tooth(params: WatchParameters, x0: float) -> Part.Shape:
    run = params.ratchet_pitch_mm * 0.65
    z0 = params.strap_thickness_mm
    points = [
        Vector(x0, -params.strap_width_mm / 2.0, z0),
        Vector(x0 + run, -params.strap_width_mm / 2.0, z0),
        Vector(x0 + run, -params.strap_width_mm / 2.0, z0 + params.ratchet_tooth_height_mm),
        Vector(x0, -params.strap_width_mm / 2.0, z0),
    ]
    face = Part.Face(Part.makePolygon(points))
    return face.extrude(Vector(0.0, params.strap_width_mm, 0.0))


def build_long_strap(params: WatchParameters) -> Part.Shape:
    body = Part.makeBox(
        params.long_strap_length_mm,
        params.strap_width_mm,
        params.strap_thickness_mm,
        Vector(-params.long_strap_length_mm, -params.strap_width_mm / 2.0, 0.0),
    )
    first_tooth_x = -params.long_strap_length_mm + 10.0
    teeth = [_ramp_tooth(params, first_tooth_x + i * params.ratchet_pitch_mm) for i in range(params.ratchet_positions)]
    return require_single_solid("Strap_Long_TPU", fuse_all([body, _hinge_tab(params, 0.0)] + teeth))


def build_short_strap(params: WatchParameters) -> Part.Shape:
    body = Part.makeBox(
        params.short_strap_length_mm,
        params.strap_width_mm,
        params.strap_thickness_mm,
        Vector(0.0, -params.strap_width_mm / 2.0, 0.0),
    )
    return require_single_solid("Strap_Short_TPU", fuse_all([body, _hinge_tab(params, 0.0)]))


def wrist_positions_mm(params: WatchParameters) -> List[float]:
    return [params.generated_wrist_min_mm + i * params.ratchet_pitch_mm for i in range(params.ratchet_positions)]
```

- [ ] **Step 4: Run FreeCADCmd tests**

Expected: exit 0, 19 positions, and no multi-solid failure from tooth fusion.

- [ ] **Step 5: Commit strap geometry**

```powershell
git add cad/freecad/xsens_dot2_watch/straps.py cad/freecad/tests/run_freecad_geometry_tests.py
git commit -m "feat: generate TPU straps and ratchet positions"
```

### Task 6: Generate the ratchet housing, pawl, and polymer pin

**Files:**
- Create: `cad/freecad/xsens_dot2_watch/ratchet.py`
- Modify: `cad/freecad/tests/run_freecad_geometry_tests.py`

- [ ] **Step 1: Add failing ratchet-component assertions**

Place the import beside the existing package imports and the function before the final invocation block.

```python
from xsens_dot2_watch.ratchet import build_pawl, build_pin, build_ratchet_housing


def test_ratchet_components():
    components = {
        "housing": build_ratchet_housing(PARAMS),
        "pawl": build_pawl(PARAMS),
        "pin": build_pin(PARAMS),
    }
    for name, shape in components.items():
        if not shape.isValid() or len(shape.Solids) != 1 or shape.Volume <= 0:
            raise AssertionError("%s must be one valid solid" % name)
    assert_close(components["housing"].BoundBox.XLength, PARAMS.ratchet_housing_length_mm)
    assert_close(components["housing"].BoundBox.YLength, PARAMS.ratchet_housing_width_mm)
    assert_close(components["pin"].BoundBox.YLength, PARAMS.ratchet_housing_width_mm)


test_ratchet_components()
```

- [ ] **Step 2: Verify the missing-module failure**

Run the FreeCADCmd test command. Expected: traceback naming `xsens_dot2_watch.ratchet`.

- [ ] **Step 3: Implement printable ratchet components**

```python
from __future__ import annotations

import Part
from FreeCAD import Vector

from .config import WatchParameters
from .geometry import fuse_all, require_single_solid


def build_ratchet_housing(params: WatchParameters) -> Part.Shape:
    outer = Part.makeBox(
        params.ratchet_housing_length_mm,
        params.ratchet_housing_width_mm,
        params.ratchet_housing_height_mm,
        Vector(0.0, -params.ratchet_housing_width_mm / 2.0, 0.0),
    )
    tunnel_width = params.strap_width_mm + 2.0 * params.ratchet_slide_clearance_per_side_mm
    tunnel_height = params.strap_thickness_mm + params.ratchet_tooth_height_mm + 0.7
    tunnel = Part.makeBox(
        params.ratchet_housing_length_mm + 2.0,
        tunnel_width,
        tunnel_height,
        Vector(-1.0, -tunnel_width / 2.0, 1.0),
    )
    release = Part.makeBox(12.0, 10.0, 4.0, Vector(8.0, -5.0, 4.0))
    pin_radius = (params.pin_diameter_mm + params.pin_hole_clearance_mm) / 2.0
    pin_hole = Part.makeCylinder(
        pin_radius,
        params.ratchet_housing_width_mm + 2.0,
        Vector(14.0, -params.ratchet_housing_width_mm / 2.0 - 1.0, 5.5),
        Vector(0.0, 1.0, 0.0),
    )
    return require_single_solid("Ratchet_Housing_PETG", outer.cut(tunnel.fuse(release).fuse(pin_hole)))


def build_pawl(params: WatchParameters) -> Part.Shape:
    lever = Part.makeBox(18.0, 8.0, 1.8, Vector(0.0, -4.0, 0.0))
    tooth_points = [
        Vector(6.0, -4.0, 0.0),
        Vector(9.0, -4.0, 0.0),
        Vector(9.0, -4.0, -params.ratchet_tooth_height_mm),
        Vector(6.0, -4.0, 0.0),
    ]
    tooth = Part.Face(Part.makePolygon(tooth_points)).extrude(Vector(0.0, 8.0, 0.0))
    pad = Part.makeBox(6.0, 10.0, 2.2, Vector(12.0, -5.0, 0.0))
    return require_single_solid("Ratchet_Pawl_PETG", fuse_all([lever, tooth, pad]))


def build_pin(params: WatchParameters) -> Part.Shape:
    pin = Part.makeCylinder(
        params.pin_diameter_mm / 2.0,
        params.ratchet_housing_width_mm,
        Vector(0.0, -params.ratchet_housing_width_mm / 2.0, 0.0),
        Vector(0.0, 1.0, 0.0),
    )
    return require_single_solid("Polymer_Pin_PETG", pin)
```

- [ ] **Step 4: Run FreeCADCmd tests**

Expected: exit 0 and no invalid-solid diagnostics.

- [ ] **Step 5: Commit ratchet geometry**

```powershell
git add cad/freecad/xsens_dot2_watch/ratchet.py cad/freecad/tests/run_freecad_geometry_tests.py
git commit -m "feat: generate printable ratchet clasp parts"
```

### Task 7: Assemble, validate, and export the complete build

**Files:**
- Create: `cad/freecad/xsens_dot2_watch/assembly.py`
- Create: `cad/freecad/xsens_dot2_watch/validation.py`
- Create: `cad/freecad/xsens_dot2_watch/exporter.py`
- Create: `cad/freecad/xsens_dot2_watch/build.py`
- Create: `cad/freecad/Build_Xsens_DOT2_Watch.FCMacro`
- Modify: `cad/freecad/tests/run_freecad_geometry_tests.py`

- [ ] **Step 1: Extend the FreeCAD test runner with assembly and export assertions**

Move the new standard-library and package imports to the existing import section; add the function before the final invocation block.

```python
import tempfile

from xsens_dot2_watch.assembly import build_components, create_assembly_document
from xsens_dot2_watch.exporter import export_build
from xsens_dot2_watch.validation import validate_components, validate_sensor_fit


def test_assembly_and_exports():
    components = build_components(PARAMS)
    validate_components(components)
    validate_sensor_fit(components)
    doc = create_assembly_document(components)
    with tempfile.TemporaryDirectory() as tmp:
        result = export_build(doc, components, Path(tmp), PARAMS)
        required = [
            result["fcstd"],
            result["assembly_step"],
            result["import_csv"],
        ]
        for path in required:
            if not Path(path).is_file():
                raise AssertionError("missing export %s" % path)
        if len(result["component_steps"]) != 8:
            raise AssertionError("expected eight unique component STEP files")
        if len(result["stl_files"]) != 7:
            raise AssertionError("expected seven unique printable STL files")


test_assembly_and_exports()
```

Eight STEP files means the seven unique printable models plus the non-print sensor dummy. Seven printable STL files means cradle, fit coupon, long strap, short strap, housing, pawl, and one reusable pin model. The pin is instantiated twice in the assembly but exported once.

- [ ] **Step 2: Verify the new test fails on missing assembly modules**

Run the FreeCADCmd test command. Expected: traceback naming `xsens_dot2_watch.assembly`.

- [ ] **Step 3: Implement component metadata and zero-rotation placements**

Use a `Component` record with `name`, `material`, `shape`, `translation_mm`, `printable`, `fit_check`, and `print_config`. Build these instances:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import FreeCAD as App

from .config import WatchParameters
from .cradle import build_cradle, build_fit_coupon
from .ratchet import build_pawl, build_pin, build_ratchet_housing
from .sensor import build_sensor_dummy
from .straps import build_long_strap, build_short_strap


@dataclass
class Component:
    name: str
    material: str
    shape: Any
    translation_mm: Tuple[float, float, float]
    printable: bool
    fit_check: bool
    print_config: bool


def build_components(params: WatchParameters) -> List[Component]:
    lug_x = params.cradle_length_mm / 2.0 + 4.0
    housing_x = lug_x + params.short_strap_length_mm
    return [
        Component("DOT2_Cradle_PETG", params.petg_material, build_cradle(params), (0.0, 0.0, 0.0), True, True, True),
        Component("Strap_Long_TPU", params.tpu_material, build_long_strap(params), (-lug_x, 0.0, 0.0), True, True, True),
        Component("Strap_Short_TPU", params.tpu_material, build_short_strap(params), (lug_x, 0.0, 0.0), True, True, True),
        Component("Ratchet_Housing_PETG", params.petg_material, build_ratchet_housing(params), (housing_x, 0.0, 0.0), True, True, True),
        Component("Ratchet_Pawl_PETG", params.petg_material, build_pawl(params), (housing_x + 5.0, 0.0, 5.0), True, True, True),
        Component("Polymer_Pin_PETG", params.petg_material, build_pin(params), (housing_x + 14.0, 0.0, 5.5), True, True, True),
        Component("Polymer_Pin_PETG_2", params.petg_material, build_pin(params), (-lug_x, 0.0, 2.8), False, True, True),
        Component("DOT2_Fit_Dummy", "REFERENCE", build_sensor_dummy(params), (0.0, 0.0, params.cradle_base_thickness_mm), False, True, False),
        Component("DOT2_Fit_Coupon_PETG", params.petg_material, build_fit_coupon(params), (0.0, 60.0, 0.0), True, False, False),
    ]


def create_assembly_document(components: List[Component]):
    doc = App.newDocument("Xsens_DOT2_Watch")
    fit_group = doc.addObject("App::Part", "FIT_CHECK")
    for component in components:
        obj = doc.addObject("Part::Feature", component.name)
        obj.Label = component.name
        obj.Shape = component.shape.copy()
        obj.addProperty("App::PropertyString", "Material", "Build")
        obj.Material = component.material
        obj.Placement.Base = App.Vector(*component.translation_mm)
        if component.fit_check:
            fit_group.addObject(obj)
    doc.recompute()
    return doc
```

Because one FreeCAD object cannot belong to two `App::Part` containers simultaneously, keep `FIT_CHECK` as the positioned assembly group and represent `PRINT` membership through the `print_config` metadata and manifest. Do not duplicate solids merely to mimic SolidWorks configurations.

- [ ] **Step 4: Implement geometry and import validation**

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import Mesh
import Part
from FreeCAD import Vector

from .assembly import Component


def validate_components(components: Iterable[Component]) -> None:
    for component in components:
        shape = component.shape
        if shape.isNull() or not shape.isValid() or len(shape.Solids) != 1 or shape.Volume <= 0:
            raise RuntimeError("invalid component: %s" % component.name)


def validate_sensor_fit(components: Iterable[Component]) -> None:
    by_name = {item.name: item for item in components}
    cradle = by_name["DOT2_Cradle_PETG"].shape.copy()
    sensor = by_name["DOT2_Fit_Dummy"].shape.copy()
    tx, ty, tz = by_name["DOT2_Fit_Dummy"].translation_mm
    sensor.translate(Vector(tx, ty, tz))
    if cradle.common(sensor).Volume > 1e-5:
        raise RuntimeError("sensor dummy has unintended solid interference with cradle")


def _dimensions(bound_box) -> Tuple[float, float, float]:
    return (bound_box.XLength, bound_box.YLength, bound_box.ZLength)


def _assert_dimensions(label: str, actual, expected, tolerance: float) -> None:
    for actual_value, expected_value in zip(actual, expected):
        if abs(actual_value - expected_value) > tolerance:
            raise RuntimeError("%s bounding-box mismatch: %r != %r" % (label, actual, expected))


def validate_step(path: Path, expected_shape: Optional[Part.Shape] = None) -> None:
    shape = Part.read(str(path))
    if shape.isNull() or not shape.isValid() or len(shape.Solids) < 1:
        raise RuntimeError("invalid STEP re-import: %s" % path)
    if expected_shape is not None:
        _assert_dimensions(str(path), _dimensions(shape.BoundBox), _dimensions(expected_shape.BoundBox), 0.01)


def validate_stl(path: Path, expected_shape: Optional[Part.Shape] = None) -> None:
    mesh = Mesh.Mesh(str(path))
    if mesh.CountFacets <= 0 or not mesh.isSolid():
        raise RuntimeError("STL is empty or open: %s" % path)
    if expected_shape is not None:
        _assert_dimensions(str(path), _dimensions(mesh.BoundBox), _dimensions(expected_shape.BoundBox), 0.10)
```

- [ ] **Step 5: Implement FCStd, STEP, STL, and CSV exports**

`solidworks_import_manifest.csv` must use this exact header:

```text
component_name,step_path,part_filename,x_mm,y_mm,z_mm,include_fit_check,include_print
```

Use these export rules:

```python
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

import FreeCAD as App
import MeshPart
import Part

from .assembly import Component
from .config import WatchParameters
from .validation import validate_step, validate_stl


def _feature(doc, name, shape):
    obj = doc.addObject("Part::Feature", name)
    obj.Shape = shape.copy()
    return obj


def export_build(doc, components: List[Component], output_dir: Path, params: WatchParameters) -> Dict[str, object]:
    output_dir = Path(output_dir)
    step_dir = output_dir / "step" / "components"
    petg_dir = output_dir / "stl" / "petg"
    tpu_dir = output_dir / "stl" / "tpu"
    for path in (step_dir, petg_dir, tpu_dir):
        path.mkdir(parents=True, exist_ok=True)

    fcstd = output_dir / "xsens_dot2_watch.FCStd"
    doc.recompute()
    doc.saveAs(str(fcstd))

    component_steps = []
    stl_files = []
    csv_rows = []
    exported_pin = False
    assembly_objects = []
    for component in components:
        if component.name == "Polymer_Pin_PETG_2":
            shared_step = step_dir / "Polymer_Pin_PETG.step"
            csv_rows.append(
                [component.name, str(shared_step.relative_to(output_dir)).replace("\\", "/"), "Polymer_Pin_PETG.SLDPRT"]
                + list(component.translation_mm)
                + [str(component.fit_check), str(component.print_config)]
            )
            continue
        if component.name == "DOT2_Fit_Dummy" or component.fit_check or component.printable:
            part_doc = App.newDocument("Export_%s" % component.name)
            obj = _feature(part_doc, component.name, component.shape)
            part_doc.recompute()
            step_path = step_dir / (component.name + ".step")
            Part.export([obj], str(step_path))
            validate_step(step_path, component.shape)
            component_steps.append(str(step_path))
            if component.fit_check or component.name == "DOT2_Fit_Dummy":
                csv_rows.append(
                    [component.name, str(step_path.relative_to(output_dir)).replace("\\", "/"), component.name + ".SLDPRT"]
                    + list(component.translation_mm)
                    + [str(component.fit_check), str(component.print_config)]
                )
            App.closeDocument(part_doc.Name)
        if component.printable:
            if component.name == "Polymer_Pin_PETG" and exported_pin:
                continue
            material_dir = tpu_dir if component.material == params.tpu_material else petg_dir
            stl_path = material_dir / (component.name + ".stl")
            mesh = MeshPart.meshFromShape(
                Shape=component.shape,
                LinearDeflection=params.stl_linear_deflection_mm,
                AngularDeflection=math.radians(params.stl_angular_deflection_deg),
                Relative=False,
            )
            mesh.write(str(stl_path))
            validate_stl(stl_path, component.shape)
            stl_files.append(str(stl_path))
            exported_pin = exported_pin or component.name == "Polymer_Pin_PETG"
    for obj in doc.Objects:
        if (
            hasattr(obj, "Shape")
            and not obj.Shape.isNull()
            and obj.Name != "DOT2_Fit_Coupon_PETG"
        ):
            assembly_objects.append(obj)
    assembly_step = output_dir / "step" / "Xsens_DOT2_Watch_Assembly.step"
    Part.export(assembly_objects, str(assembly_step))
    validate_step(assembly_step)
    import_csv = output_dir / "solidworks_import_manifest.csv"
    with import_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component_name", "step_path", "part_filename", "x_mm", "y_mm", "z_mm", "include_fit_check", "include_print"])
        writer.writerows(csv_rows)
    return {
        "fcstd": str(fcstd),
        "assembly_step": str(assembly_step),
        "import_csv": str(import_csv),
        "component_steps": component_steps,
        "stl_files": stl_files,
    }
```

Record `stl_linear_deflection_mm` and `stl_angular_deflection_deg` from `params.to_dict()` in `parameters.json` and `manifest.json`; the `MeshPart.meshFromShape` call above is the authoritative tessellation path.

- [ ] **Step 6: Implement orchestration and the thin `.FCMacro` entry point**

`build.py` must read `XSENS_DOT2_CONFIG` and `XSENS_DOT2_OUTPUT_ROOT`, create a new timestamped run, log `READ/WRITE/SKIP/FAIL`, write `parameters.json`, run geometry/export validation, hash every artifact, and finalize `manifest.json` with `status: complete` only after all checks pass.

Use this orchestration contract:

```python
from __future__ import annotations

import os
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

import FreeCAD as App

from .artifacts import BuildLog, create_run_directory, sha256_file, write_json
from .assembly import build_components, create_assembly_document
from .config import load_parameters
from .exporter import export_build
from .validation import validate_components, validate_sensor_fit


def _git_commit(repo: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(environ: Mapping[str, str]) -> int:
    config_value = environ.get("XSENS_DOT2_CONFIG")
    output_value = environ.get("XSENS_DOT2_OUTPUT_ROOT")
    if not config_value or not output_value:
        print("FAIL environment reason=XSENS_DOT2_CONFIG and XSENS_DOT2_OUTPUT_ROOT are required")
        return 1
    config_path = Path(config_value).resolve()
    output_root = Path(output_value).resolve()
    run_dir = create_run_directory(output_root, datetime.now())
    log = BuildLog(run_dir / "build_log.txt")
    manifest_path = run_dir / "manifest.json"
    repo = config_path.parents[2]
    generator_path = Path(__file__).resolve().parents[1] / "Build_Xsens_DOT2_Watch.FCMacro"
    manifest = {
        "status": "building",
        "git_commit": _git_commit(repo),
        "freecad_version": list(App.Version()),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "generator_sha256": sha256_file(generator_path),
        "artifacts": [],
    }
    try:
        log.read(str(config_path))
        params = load_parameters(config_path)
        parameters_path = run_dir / "parameters.json"
        write_json(parameters_path, params.to_dict())
        log.write(str(parameters_path))
        components = build_components(params)
        validate_components(components)
        validate_sensor_fit(components)
        log.skip("Polymer_Pin_PETG_2.stl", "second assembly pin reuses the first pin STL")
        log.skip("DOT2_Fit_Dummy.stl", "reference component is not printable")
        log.skip("DOT2_Fit_Coupon_PETG in assembly", "fit coupon is calibration-only")
        manifest["components"] = [
            {
                "name": item.name,
                "material": item.material,
                "translation_mm": list(item.translation_mm),
                "bounding_box_mm": [
                    item.shape.BoundBox.XLength,
                    item.shape.BoundBox.YLength,
                    item.shape.BoundBox.ZLength,
                ],
                "printable": item.printable,
                "fit_check": item.fit_check,
                "print_config": item.print_config,
            }
            for item in components
        ]
        doc = create_assembly_document(components)
        result = export_build(doc, components, run_dir, params)
        App.closeDocument(doc.Name)
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or path.name in {"build_log.txt", "manifest.json"}:
                continue
            manifest["artifacts"].append(
                {
                    "relative_path": path.relative_to(run_dir).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
            log.write(str(path))
        manifest["status"] = "complete"
        manifest["parameters"] = params.to_dict()
        manifest["exports"] = result
        write_json(manifest_path, manifest)
        log.write(str(manifest_path))
        print("BUILD_COMPLETE run_dir=%s" % run_dir)
        return 0
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = "%s: %s" % (type(exc).__name__, exc)
        manifest["traceback"] = traceback.format_exc()
        write_json(manifest_path, manifest)
        log.fail("build", manifest["error"])
        print("BUILD_FAILED run_dir=%s error=%s" % (run_dir, manifest["error"]))
        return 1
```

The artifact hash list intentionally excludes `manifest.json` to avoid recursive self-hashing and excludes `build_log.txt` because logging continues through manifest finalization.

The entry point is exactly:

```python
from __future__ import print_function

import os
import sys
from pathlib import Path

MACRO_PATH = Path(__file__).resolve()
PACKAGE_ROOT = MACRO_PATH.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from xsens_dot2_watch.build import main

raise SystemExit(main(os.environ))
```

The `main` function returns 0 for a complete build and 1 after logging a failed stage.

- [ ] **Step 7: Run FreeCAD geometry and export tests**

Run:

```powershell
& 'E:\FreeCAD\bin\FreeCADCmd.exe' -P "$PWD\cad\freecad" "$PWD\cad\freecad\tests\run_freecad_geometry_tests.py"
```

Expected: exit 0, output contains `FREECAD_TEST_PASS`, eight unique component STEP files, and seven unique printable STL files in the temporary directory.

- [ ] **Step 8: Commit assembly and export pipeline**

```powershell
git add cad/freecad/Build_Xsens_DOT2_Watch.FCMacro cad/freecad/xsens_dot2_watch cad/freecad/tests/run_freecad_geometry_tests.py
git commit -m "feat: assemble validate and export the DOT watch"
```

### Task 8: Add the repeatable CLI wrapper and artifact verifier

**Files:**
- Create: `scripts/build_xsens_dot2_watch.ps1`
- Create: `cad/freecad/tests/verify_build.py`

- [ ] **Step 1: Add a host-Python verifier before writing the wrapper**

```python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise SystemExit("manifest status is not complete")
    for item in manifest["artifacts"]:
        path = args.run_dir / item["relative_path"]
        if not path.is_file():
            raise SystemExit("missing artifact: %s" % path)
        if digest(path) != item["sha256"]:
            raise SystemExit("hash mismatch: %s" % path)
    print("BUILD_VERIFY_PASS artifacts=%d" % len(manifest["artifacts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Implement the PowerShell wrapper with explicit read/write/skip output**

```powershell
[CmdletBinding()]
param(
    [string]$FreeCADCmd = 'E:\FreeCAD\bin\FreeCADCmd.exe',
    [string]$Config = 'configs\cad\xsens_dot2_watch.json',
    [string]$OutputRoot = 'outputs\cad\xsens_dot2_watch'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$freecad = (Resolve-Path -LiteralPath $FreeCADCmd).Path
$configPath = (Resolve-Path -LiteralPath (Join-Path $repo $Config)).Path
$macroPath = (Resolve-Path -LiteralPath (Join-Path $repo 'cad\freecad\Build_Xsens_DOT2_Watch.FCMacro')).Path
$outputPath = Join-Path $repo $OutputRoot

Write-Host "READ freecad=$freecad"
Write-Host "READ config=$configPath"
Write-Host "READ macro=$macroPath"
Write-Host "WRITE output_root=$outputPath"

$env:XSENS_DOT2_CONFIG = $configPath
$env:XSENS_DOT2_OUTPUT_ROOT = $outputPath
try {
    & $freecad -P (Join-Path $repo 'cad\freecad') $macroPath
    if ($LASTEXITCODE -ne 0) {
        throw "FreeCAD generator failed with exit code $LASTEXITCODE"
    }
} finally {
    Remove-Item Env:XSENS_DOT2_CONFIG -ErrorAction SilentlyContinue
    Remove-Item Env:XSENS_DOT2_OUTPUT_ROOT -ErrorAction SilentlyContinue
}

$latest = Get-ChildItem -LiteralPath $outputPath -Directory | Sort-Object Name -Descending | Select-Object -First 1
if (-not $latest) {
    throw 'Generator completed without a timestamped output directory'
}
Write-Host "WRITE run_dir=$($latest.FullName)"
uv run python (Join-Path $repo 'cad\freecad\tests\verify_build.py') $latest.FullName
if ($LASTEXITCODE -ne 0) {
    throw "Artifact verification failed with exit code $LASTEXITCODE"
}
```

- [ ] **Step 3: Run the first persistent end-to-end build**

Run:

```powershell
& .\scripts\build_xsens_dot2_watch.ps1
```

Expected: console contains each `READ` and `WRITE`, the FreeCAD build exits 0, and verification prints `BUILD_VERIFY_PASS`. The generated directory is under `outputs/cad/xsens_dot2_watch/<timestamp>/` and remains ignored by git.

- [ ] **Step 4: Verify a second run creates a new directory**

Run the wrapper again. Expected: two timestamped directories exist and no file in the first directory changes.

- [ ] **Step 5: Commit the wrapper and verifier**

```powershell
git add scripts/build_xsens_dot2_watch.ps1 cad/freecad/tests/verify_build.py
git commit -m "feat: add repeatable FreeCAD watch build command"
```

### Task 9: Add the SolidWorks 2022 import macro and static contract tests

**Files:**
- Create: `cad/solidworks/Import_Xsens_DOT2_Watch.bas`
- Create: `tests/test_xsens_dot2_solidworks_macro.py`

- [ ] **Step 1: Write a failing static contract test**

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MACRO = ROOT / "cad" / "solidworks" / "Import_Xsens_DOT2_Watch.bas"


def test_solidworks_macro_has_required_api_contract() -> None:
    text = MACRO.read_text(encoding="utf-8")
    required = [
        "Sub main()",
        "GetCurrentMacroPathName",
        "GetImportFileData",
        "LoadFile4",
        "ImportDiagnosis",
        "Extension.SaveAs",
        "AddComponent5",
        "AddConfiguration3",
        "SetComponentState",
        "FIT_CHECK",
        "PRINT",
        "READ ",
        "WRITE ",
        "SKIP ",
        "FAIL ",
    ]
    for token in required:
        assert token in text
    assert "E:\\" not in text
    assert "C:\\" not in text
```

- [ ] **Step 2: Run the test and verify the missing-file failure**

Run `uv run pytest tests/test_xsens_dot2_solidworks_macro.py -q`.

Expected: FAIL with `FileNotFoundError` for the VBA module.

- [ ] **Step 3: Implement the VBA import sequence using the generated CSV**

The module must:

1. use `Application.SldWorks` and keep SolidWorks user-controlled;
2. prompt for the build directory containing `solidworks_import_manifest.csv`;
3. open the log in that build directory;
4. import each STEP using `GetImportFileData` and `LoadFile4`;
5. run `ImportDiagnosis(True, False, True, 0)`;
6. save each imported document using `ModelDocExtension.SaveAs`;
7. create a new assembly using the user's default assembly template;
8. open each saved part before `AddComponent5` and insert it at CSV XYZ converted from mm to m;
9. create `FIT_CHECK` and `PRINT` with `AddConfiguration3`;
10. suppress `DOT2_Fit_Dummy` from `PRINT` with `SetComponentState` and leave the fit coupon out of both assembly configurations;
11. save `Xsens_DOT2_Watch.SLDASM` and write all `READ/WRITE/SKIP/FAIL` records.

Use this concrete save/import core:

```vb
Private Function ImportStepAsPart(ByVal stepPath As String, ByVal partPath As String) As SldWorks.ModelDoc2
    Dim importData As SldWorks.ImportStepData
    Dim imported As SldWorks.ModelDoc2
    Dim partDoc As SldWorks.PartDoc
    Dim errors As Long
    Dim warnings As Long
    Dim ok As Boolean

    WriteLog "READ " & stepPath
    Set importData = swApp.GetImportFileData(stepPath)
    importData.MapConfigurationData = True
    Set imported = swApp.LoadFile4(stepPath, "r", importData, errors)
    If imported Is Nothing Or errors <> 0 Then
        WriteLog "FAIL import_step errors=" & CStr(errors) & " path=" & stepPath
        Err.Raise vbObjectError + 100, "ImportStepAsPart", "STEP import failed: " & stepPath
    End If
    Set partDoc = imported
    errors = partDoc.ImportDiagnosis(True, False, True, 0)
    If errors <> 0 Then
        WriteLog "FAIL import_diagnosis errors=" & CStr(errors) & " path=" & stepPath
        Err.Raise vbObjectError + 101, "ImportStepAsPart", "Import diagnosis failed: " & stepPath
    End If
    ok = imported.Extension.SaveAs(partPath, 0, swSaveAsOptions_Silent, Nothing, errors, warnings)
    If Not ok Or errors <> 0 Then
        WriteLog "FAIL save_part errors=" & CStr(errors) & " warnings=" & CStr(warnings) & " path=" & partPath
        Err.Raise vbObjectError + 102, "ImportStepAsPart", "Part save failed: " & partPath
    End If
    WriteLog "WRITE " & partPath
    Set ImportStepAsPart = imported
End Function
```

Use these module-level declarations and control flow around that core:

```vb
Option Explicit

Private swApp As SldWorks.SldWorks
Private logPath As String

Sub main()
    Dim macroPath As String
    Dim macroDir As String
    Dim buildRoot As String
    Dim manifestPath As String
    Dim nativeDir As String
    Dim rows As Collection
    Dim row As Variant
    Dim imported As SldWorks.ModelDoc2
    Dim partPaths As New Collection
    Dim assemblyModel As SldWorks.ModelDoc2
    Dim assemblyDoc As SldWorks.AssemblyDoc
    Dim component As SldWorks.Component2
    Dim dummyComponent As SldWorks.Component2
    Dim templatePath As String
    Dim assemblyPath As String
    Dim saveErrors As Long
    Dim saveWarnings As Long
    Dim ok As Boolean
    Dim dummyComponents(0) As Object

    On Error GoTo Failed
    Set swApp = Application.SldWorks
    swApp.Visible = True
    swApp.UserControl = True
    macroPath = swApp.GetCurrentMacroPathName
    macroDir = Left$(macroPath, InStrRev(macroPath, "\\") - 1)
    buildRoot = InputBox("Select the completed FreeCAD build directory", "Xsens DOT 2 Watch", macroDir)
    If Len(buildRoot) = 0 Then Exit Sub
    If Right$(buildRoot, 1) = "\\" Then buildRoot = Left$(buildRoot, Len(buildRoot) - 1)
    manifestPath = buildRoot & "\\solidworks_import_manifest.csv"
    If Len(Dir$(manifestPath)) = 0 Then Err.Raise vbObjectError + 1, "main", "Missing " & manifestPath
    logPath = buildRoot & "\\solidworks_build_log.txt"
    WriteLog "READ " & manifestPath
    Set rows = ReadManifest(manifestPath)
    nativeDir = buildRoot & "\\solidworks"
    EnsureFolder nativeDir
    assemblyPath = nativeDir & "\\Xsens_DOT2_Watch.SLDASM"
    If Len(Dir$(assemblyPath)) > 0 Then
        WriteLog "SKIP " & assemblyPath & " reason=already exists"
        Err.Raise vbObjectError + 7, "main", "Assembly already exists; choose a new FreeCAD build directory"
    End If

    For Each row In rows
        Dim stepPath As String
        Dim partPath As String
        stepPath = buildRoot & "\\" & Replace(CStr(row(1)), "/", "\\")
        partPath = nativeDir & "\\" & CStr(row(2))
        If Len(Dir$(partPath)) > 0 Then
            WriteLog "SKIP " & partPath & " reason=already exists"
        Else
            Set imported = ImportStepAsPart(stepPath, partPath)
            swApp.CloseDoc imported.GetTitle
        End If
        partPaths.Add Array(CStr(row(0)), partPath, CDbl(row(3)), CDbl(row(4)), CDbl(row(5)), LCase$(CStr(row(6))) = "true", LCase$(CStr(row(7))) = "true")
    Next row

    templatePath = swApp.GetUserPreferenceStringValue(swDefaultTemplateAssembly)
    If Len(templatePath) = 0 Then Err.Raise vbObjectError + 2, "main", "Default assembly template is not configured"
    Set assemblyModel = swApp.NewDocument(templatePath, swDwgPaperA4size, 0#, 0#)
    If assemblyModel Is Nothing Then Err.Raise vbObjectError + 3, "main", "Cannot create assembly document"
    Set assemblyDoc = assemblyModel

    For Each row In partPaths
        Set component = AddImportedComponent(assemblyModel, assemblyDoc, CStr(row(1)), CDbl(row(2)), CDbl(row(3)), CDbl(row(4)))
        If CStr(row(0)) = "DOT2_Fit_Dummy" Then Set dummyComponent = component
    Next row

    assemblyModel.AddConfiguration3 "FIT_CHECK", "Sensor fit inspection", "", 0
    assemblyModel.AddConfiguration3 "PRINT", "Printable assembly without sensor dummy", "", 0
    If dummyComponent Is Nothing Then Err.Raise vbObjectError + 4, "main", "DOT2_Fit_Dummy was not inserted"
    Set dummyComponents(0) = dummyComponent
    ok = assemblyDoc.SetComponentState(swComponentSuppressed, dummyComponents, swSpecifyConfiguration, "PRINT", False)
    If Not ok Then Err.Raise vbObjectError + 5, "main", "Cannot suppress DOT2_Fit_Dummy in PRINT"
    assemblyModel.ShowConfiguration2 "FIT_CHECK"
    ok = assemblyModel.Extension.SaveAs(assemblyPath, 0, swSaveAsOptions_Silent, Nothing, saveErrors, saveWarnings)
    If Not ok Or saveErrors <> 0 Then Err.Raise vbObjectError + 6, "main", "Assembly save failed"
    WriteLog "WRITE " & assemblyPath
    MsgBox "SolidWorks assembly created:" & vbCrLf & assemblyPath, vbInformation
    Exit Sub

Failed:
    WriteLog "FAIL main error=" & CStr(Err.Number) & " reason=" & Err.Description
    MsgBox Err.Description, vbCritical
End Sub


Private Function ReadManifest(ByVal path As String) As Collection
    Dim result As New Collection
    Dim handle As Integer
    Dim line As String
    Dim fields As Variant
    handle = FreeFile
    Open path For Input As #handle
    Line Input #handle, line
    Do While Not EOF(handle)
        Line Input #handle, line
        If Len(Trim$(line)) > 0 Then
            fields = Split(line, ",")
            If UBound(fields) <> 7 Then
                Close #handle
                Err.Raise vbObjectError + 20, "ReadManifest", "Invalid CSV row: " & line
            End If
            result.Add fields
        End If
    Loop
    Close #handle
    Set ReadManifest = result
End Function


Private Function AddImportedComponent(ByVal assemblyModel As SldWorks.ModelDoc2, ByVal assemblyDoc As SldWorks.AssemblyDoc, ByVal partPath As String, ByVal xMm As Double, ByVal yMm As Double, ByVal zMm As Double) As SldWorks.Component2
    Dim errors As Long
    Dim warnings As Long
    Dim partModel As SldWorks.ModelDoc2
    Set partModel = swApp.OpenDoc6(partPath, swDocPART, swOpenDocOptions_Silent, "", errors, warnings)
    If partModel Is Nothing Or errors <> 0 Then Err.Raise vbObjectError + 30, "AddImportedComponent", "Cannot open " & partPath
    swApp.ActivateDoc3 assemblyModel.GetTitle, True, swUserDecision, errors
    Set AddImportedComponent = assemblyDoc.AddComponent5(partPath, swAddComponentConfigOptions_CurrentSelectedConfig, "", False, "", xMm / 1000#, yMm / 1000#, zMm / 1000#)
    If AddImportedComponent Is Nothing Then Err.Raise vbObjectError + 31, "AddImportedComponent", "Cannot insert " & partPath
    WriteLog "READ " & partPath
End Function


Private Sub EnsureFolder(ByVal path As String)
    If Len(Dir$(path, vbDirectory)) = 0 Then
        MkDir path
        WriteLog "WRITE " & path
    Else
        WriteLog "SKIP " & path & " reason=directory exists"
    End If
End Sub


Private Sub WriteLog(ByVal message As String)
    Dim handle As Integer
    If Len(logPath) = 0 Then Exit Sub
    handle = FreeFile
    Open logPath For Append As #handle
    Print #handle, Format$(Now, "yyyy-mm-dd hh:nn:ss") & " " & message
    Close #handle
    Debug.Print message
End Sub
```

The CSV contains paths relative to the selected build directory, so Chinese characters in the workspace root never pass through the CSV parser. `swApp.GetCurrentMacroPathName` supplies only the default `InputBox` location and never hard-codes a build output.

- [ ] **Step 4: Run the static contract test**

Run:

```powershell
uv run pytest tests/test_xsens_dot2_solidworks_macro.py -q
```

Expected: 1 passed. This is not a runtime SolidWorks validation.

- [ ] **Step 5: Commit the SolidWorks bridge**

```powershell
git add cad/solidworks/Import_Xsens_DOT2_Watch.bas tests/test_xsens_dot2_solidworks_macro.py
git commit -m "feat: add SolidWorks 2022 STEP assembly macro"
```

### Task 10: Document operation and perform final verification

**Files:**
- Create: `docs/CAD_XSENS_DOT2_WATCH.md`

- [ ] **Step 1: Write the user guide with exact commands and calibration order**

The guide must contain:

````markdown
# Xsens DOT 2 Adjustable Watch CAD

## Generate with FreeCAD 0.21.1

```powershell
& .\scripts\build_xsens_dot2_watch.ps1
```

The command reads `configs/cad/xsens_dot2_watch.json` and writes a new timestamped directory under `outputs/cad/xsens_dot2_watch/`. It never overwrites a previous build.

## Print order

1. Print `DOT2_Fit_Coupon_PETG.stl`.
2. Measure the physical DOT 2 and adjust `sensor_corner_radius_mm` or `xy_clearance_per_side_mm` if required.
3. Print `Ratchet_Pawl_PETG.stl` and a short section of `Strap_Long_TPU.stl`; cycle the release before printing the remaining parts.
4. Print the cradle, both straps, housing, pawl, and two copies of the polymer pin.
5. Check the 140 mm, 176 mm, and 212 mm discrete positions before dynamic use.

## Import into SolidWorks 2022

Create a new SolidWorks VBA macro, import `cad/solidworks/Import_Xsens_DOT2_Watch.bas`, run `main`, and select the completed FreeCAD build directory. Inspect `FIT_CHECK`, `PRINT`, and `solidworks_build_log.txt` before saving downstream changes.
````

Also document PETG/TPU print orientations, the 0.4 mm nozzle and 0.20 mm layer starting assumptions, sensor removal before charging, and the absence of waterproofing or certified wear safety.

- [ ] **Step 2: Add a design-reference section to the guide**

Link `docs/superpowers/specs/2026-07-14-xsens-dot2-adjustable-watch-cad-design.md` from the guide and state that generated output paths are intentionally not used as permanent documentation links.

- [ ] **Step 3: Run all focused checks**

```powershell
uv run pytest tests/test_xsens_dot2_watch_config.py tests/test_xsens_dot2_watch_artifacts.py tests/test_xsens_dot2_solidworks_macro.py -q
uv run ruff check tests/test_xsens_dot2_watch_config.py tests/test_xsens_dot2_watch_artifacts.py tests/test_xsens_dot2_solidworks_macro.py
uv run ruff check --target-version py38 cad/freecad/xsens_dot2_watch cad/freecad/tests
& 'E:\FreeCAD\bin\FreeCADCmd.exe' -P "$PWD\cad\freecad" "$PWD\cad\freecad\tests\run_freecad_geometry_tests.py"
& .\scripts\build_xsens_dot2_watch.ps1
```

Expected: pytest and ruff exit 0, FreeCAD prints `FREECAD_TEST_PASS`, and the wrapper prints `BUILD_VERIFY_PASS`.

- [ ] **Step 4: Inspect generated geometry visually in FreeCAD**

Open the latest `xsens_dot2_watch.FCStd` in FreeCAD 0.21.1 and inspect top, bottom, side, and isometric views. Confirm:

- DOT 2 top button and LED surface are unobstructed;
- the sensor dummy sits above the base without non-retention intersection;
- both snap hooks overhang the upper edge;
- strap and cradle pin holes are coaxial in the positioned assembly;
- pawl tooth faces the tightening direction;
- fit coupon is separate from the assembly;
- no printable component is represented as more than one solid.

Record any screenshots only in the ignored timestamped output directory.

- [ ] **Step 5: Confirm repository hygiene**

Run:

```powershell
git status --short
git check-ignore -v outputs/cad/xsens_dot2_watch
git diff --check
```

Expected: `outputs/cad/xsens_dot2_watch` is ignored; no generated FCStd, STEP, STL, logs, manifests, or previews are staged; `git diff --check` is clean.

- [ ] **Step 6: Commit documentation and final validated changes**

```powershell
git add docs/CAD_XSENS_DOT2_WATCH.md
git commit -m "docs: explain Xsens DOT 2 watch CAD workflow"
```

## SolidWorks Runtime Handoff

SolidWorks 2022 is not installed in the current environment, so do not claim runtime success for the VBA macro. After the user runs it, inspect `solidworks_build_log.txt`. If it reports a failure, reproduce the exact failing API stage against the SolidWorks 2022 official API documentation and patch only the VBA bridge; do not change validated FreeCAD geometry unless the imported STEP itself is invalid.
