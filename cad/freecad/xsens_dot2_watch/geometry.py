from __future__ import annotations

from collections.abc import Iterable

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
        raise RuntimeError(f"{name} is not one valid non-zero solid")
    return shape.removeSplitter()
