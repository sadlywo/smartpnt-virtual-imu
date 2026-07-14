from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    translation_mm: tuple[float, float, float]
    printable: bool
    fit_check: bool
    print_config: bool


def build_components(params: WatchParameters) -> list[Component]:
    lug_x = params.cradle_length_mm / 2.0 + 4.0
    housing_x = lug_x + params.short_strap_length_mm
    return [
        Component(
            "DOT2_Cradle_PETG",
            params.petg_material,
            build_cradle(params),
            (0.0, 0.0, 0.0),
            True,
            True,
            True,
        ),
        Component(
            "Strap_Long_TPU",
            params.tpu_material,
            build_long_strap(params),
            (-lug_x, 0.0, 0.0),
            True,
            True,
            True,
        ),
        Component(
            "Strap_Short_TPU",
            params.tpu_material,
            build_short_strap(params),
            (lug_x, 0.0, 0.0),
            True,
            True,
            True,
        ),
        Component(
            "Ratchet_Housing_PETG",
            params.petg_material,
            build_ratchet_housing(params),
            (housing_x, 0.0, 0.0),
            True,
            True,
            True,
        ),
        Component(
            "Ratchet_Pawl_PETG",
            params.petg_material,
            build_pawl(params),
            (housing_x + 5.0, 0.0, 5.0),
            True,
            True,
            True,
        ),
        Component(
            "Polymer_Pin_PETG",
            params.petg_material,
            build_pin(params),
            (-lug_x, 0.0, 2.8),
            True,
            True,
            True,
        ),
        Component(
            "Polymer_Pin_PETG_2",
            params.petg_material,
            build_pin(params),
            (lug_x, 0.0, 2.8),
            False,
            True,
            True,
        ),
        Component(
            "DOT2_Fit_Dummy",
            "REFERENCE",
            build_sensor_dummy(params),
            (0.0, 0.0, params.cradle_base_thickness_mm),
            False,
            True,
            False,
        ),
        Component(
            "DOT2_Fit_Coupon_PETG",
            params.petg_material,
            build_fit_coupon(params),
            (0.0, 60.0, 0.0),
            True,
            False,
            False,
        ),
    ]


def create_assembly_document(components: list[Component]):
    doc = App.newDocument("Xsens_DOT2_Watch")
    fit_group = doc.addObject("App::Part", "FIT_CHECK")
    for component in components:
        obj = doc.addObject("Part::Feature", component.name)
        obj.Label = component.name
        obj.Shape = component.shape.copy()
        obj.addProperty("App::PropertyString", "Material", "Build")
        obj.Material = component.material
        obj.addProperty("App::PropertyBool", "Printable", "Build")
        obj.Printable = component.printable
        obj.addProperty("App::PropertyBool", "IncludeInPrintConfig", "Build")
        obj.IncludeInPrintConfig = component.print_config
        obj.Placement.Base = App.Vector(*component.translation_mm)
        if component.fit_check:
            fit_group.addObject(obj)
    doc.recompute()
    return doc
