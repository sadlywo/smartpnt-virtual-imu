from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import Mesh
import Part
from FreeCAD import Vector

from .assembly import Component


def validate_components(components: Iterable[Component]) -> None:
    for component in components:
        shape = component.shape
        if shape.isNull() or not shape.isValid() or len(shape.Solids) != 1 or shape.Volume <= 0:
            raise RuntimeError(f"invalid component: {component.name}")


def validate_sensor_fit(components: Iterable[Component]) -> None:
    by_name = {item.name: item for item in components}
    cradle = by_name["DOT2_Cradle_PETG"].shape.copy()
    sensor = by_name["DOT2_Fit_Dummy"].shape.copy()
    tx, ty, tz = by_name["DOT2_Fit_Dummy"].translation_mm
    sensor.translate(Vector(tx, ty, tz))
    if cradle.common(sensor).Volume > 1e-5:
        raise RuntimeError("sensor dummy has unintended solid interference with cradle")


def _dimensions(bound_box) -> tuple[float, float, float]:
    return (bound_box.XLength, bound_box.YLength, bound_box.ZLength)


def _assert_dimensions(label: str, actual, expected, tolerance: float) -> None:
    for actual_value, expected_value in zip(actual, expected):
        if abs(actual_value - expected_value) > tolerance:
            raise RuntimeError(f"{label} bounding-box mismatch: {actual!r} != {expected!r}")


def validate_step(path: Path, expected_shape: Part.Shape | None = None) -> None:
    shape = Part.read(str(path))
    if shape.isNull() or not shape.isValid() or len(shape.Solids) < 1:
        raise RuntimeError(f"invalid STEP re-import: {path}")
    if expected_shape is not None:
        _assert_dimensions(
            str(path),
            _dimensions(shape.BoundBox),
            _dimensions(expected_shape.BoundBox),
            0.01,
        )


def validate_stl(path: Path, expected_shape: Part.Shape | None = None) -> None:
    mesh = Mesh.Mesh(str(path))
    if mesh.CountFacets <= 0 or not mesh.isSolid():
        raise RuntimeError(f"STL is empty or open: {path}")
    if expected_shape is not None:
        _assert_dimensions(
            str(path),
            _dimensions(mesh.BoundBox),
            _dimensions(expected_shape.BoundBox),
            0.10,
        )
