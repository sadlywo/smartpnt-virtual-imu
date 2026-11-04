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
