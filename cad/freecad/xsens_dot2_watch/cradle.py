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
    outer_radius = (
        params.sensor_corner_radius_mm
        + params.xy_clearance_per_side_mm
        + params.cradle_wall_thickness_mm
    )
    outer = rounded_box(
        outer_length,
        outer_width,
        params.cradle_low_wall_height_mm,
        outer_radius,
    )
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
        [
            base,
            ring,
            top_post,
            bottom_post,
            top_hook,
            bottom_hook,
            _lug(params, -1),
            _lug(params, 1),
        ]
    )
    return require_single_solid("DOT2_Cradle_PETG", shape)


def build_fit_coupon(params: WatchParameters) -> Part.Shape:
    cradle = build_cradle(params)
    cutter = Part.makeBox(
        20.0,
        params.cradle_width_mm / 2.0,
        20.0,
        Vector(-10.0, 0.0, 0.0),
    )
    return require_single_solid("DOT2_Fit_Coupon_PETG", cradle.common(cutter))
