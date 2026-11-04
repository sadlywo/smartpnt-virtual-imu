from __future__ import annotations

import Part
from FreeCAD import Vector

from .config import WatchParameters
from .geometry import fuse_all, require_single_solid


def _hinge_tab(params: WatchParameters, center_x: float) -> Part.Shape:
    return Part.makeBox(
        8.0,
        params.strap_width_mm,
        5.0,
        Vector(center_x - 4.0, -params.strap_width_mm / 2.0, 0.0),
    )


def _hinge_hole(params: WatchParameters, center_x: float) -> Part.Shape:
    radius = (params.pin_diameter_mm + params.pin_hole_clearance_mm) / 2.0
    return Part.makeCylinder(
        radius,
        params.strap_width_mm + 2.0,
        Vector(center_x, -params.strap_width_mm / 2.0 - 1.0, 2.8),
        Vector(0.0, 1.0, 0.0),
    )


def _ramp_tooth(params: WatchParameters, x0: float) -> Part.Shape:
    run = params.ratchet_pitch_mm * 0.65
    z0 = params.strap_thickness_mm
    points = [
        Vector(x0, -params.strap_width_mm / 2.0, z0),
        Vector(x0 + run, -params.strap_width_mm / 2.0, z0),
        Vector(
            x0 + run,
            -params.strap_width_mm / 2.0,
            z0 + params.ratchet_tooth_height_mm,
        ),
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
    teeth = [
        _ramp_tooth(params, first_tooth_x + i * params.ratchet_pitch_mm)
        for i in range(params.ratchet_positions)
    ]
    shape = fuse_all([body, _hinge_tab(params, 0.0), *teeth]).cut(
        _hinge_hole(params, 0.0)
    )
    return require_single_solid("Strap_Long_TPU", shape)


def build_short_strap(params: WatchParameters) -> Part.Shape:
    body = Part.makeBox(
        params.short_strap_length_mm,
        params.strap_width_mm,
        params.strap_thickness_mm,
        Vector(0.0, -params.strap_width_mm / 2.0, 0.0),
    )
    arm_x = params.short_strap_length_mm - 2.0
    positive_arm = Part.makeBox(10.0, 1.4, 2.0, Vector(arm_x, 9.6, 0.8))
    negative_arm = Part.makeBox(10.0, 1.4, 2.0, Vector(arm_x, -11.0, 0.8))
    positive_barb = Part.makeBox(2.0, 2.0, 2.0, Vector(arm_x + 8.0, 9.6, 0.8))
    negative_barb = Part.makeBox(2.0, 2.0, 2.0, Vector(arm_x + 8.0, -11.6, 0.8))
    shape = fuse_all(
        [
            body,
            _hinge_tab(params, 0.0),
            positive_arm,
            negative_arm,
            positive_barb,
            negative_barb,
        ]
    ).cut(_hinge_hole(params, 0.0))
    return require_single_solid("Strap_Short_TPU", shape)


def wrist_positions_mm(params: WatchParameters) -> list[float]:
    return [
        params.generated_wrist_min_mm + i * params.ratchet_pitch_mm
        for i in range(params.ratchet_positions)
    ]
