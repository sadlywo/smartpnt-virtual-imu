from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


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

    def validate(self) -> WatchParameters:
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and name != "ratchet_positions" and value <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.ratchet_positions < 2:
            raise ValueError("ratchet_positions must be >= 2")
        if self.wrist_target_min_mm >= self.wrist_target_max_mm:
            raise ValueError("wrist_target_min_mm must be smaller than wrist_target_max_mm")
        if self.minimum_overlap_mm <= 0:
            raise ValueError("ratchet field is longer than available overlap")
        if self.generated_wrist_max_mm < self.wrist_target_max_mm:
            raise ValueError("ratchet positions do not cover wrist_target_max_mm")
        return self

    def to_dict(self) -> dict[str, Any]:
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
        raise ValueError(f"Unknown parameter keys: {', '.join(unknown)}")
    return WatchParameters(**raw).validate()
