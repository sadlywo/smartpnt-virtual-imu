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
    release_window = Part.makeBox(12.0, 10.0, 4.0, Vector(8.0, -5.0, 4.0))
    return require_single_solid(
        "Ratchet_Housing_PETG",
        outer.cut(tunnel.fuse(release_window)),
    )


def build_pawl(params: WatchParameters) -> Part.Shape:
    lever = Part.makeBox(18.0, 8.0, 1.8, Vector(0.0, -4.0, 0.0))
    tooth_points = [
        Vector(6.0, -4.0, 0.0),
        Vector(9.0, -4.0, 0.0),
        Vector(9.0, -4.0, -params.ratchet_tooth_height_mm),
        Vector(6.0, -4.0, 0.0),
    ]
    tooth = Part.Face(Part.makePolygon(tooth_points)).extrude(Vector(0.0, 8.0, 0.0))
    release_pad = Part.makeBox(6.0, 10.0, 2.2, Vector(12.0, -5.0, 0.0))
    return require_single_solid(
        "Ratchet_Pawl_PETG",
        fuse_all([lever, tooth, release_pad]),
    )


def build_pin(params: WatchParameters) -> Part.Shape:
    pin = Part.makeCylinder(
        params.pin_diameter_mm / 2.0,
        params.ratchet_housing_width_mm,
        Vector(0.0, -params.ratchet_housing_width_mm / 2.0, 0.0),
        Vector(0.0, 1.0, 0.0),
    )
    return require_single_solid("Polymer_Pin_PETG", pin)
