import tempfile
from pathlib import Path

import Part
from FreeCAD import Vector
from xsens_dot2_watch.assembly import build_components, create_assembly_document
from xsens_dot2_watch.config import load_parameters
from xsens_dot2_watch.cradle import build_cradle, build_fit_coupon
from xsens_dot2_watch.exporter import export_build
from xsens_dot2_watch.geometry import rounded_box
from xsens_dot2_watch.ratchet import build_pawl, build_pin, build_ratchet_housing
from xsens_dot2_watch.sensor import build_sensor_dummy
from xsens_dot2_watch.straps import build_long_strap, build_short_strap, wrist_positions_mm
from xsens_dot2_watch.validation import validate_components, validate_sensor_fit

ROOT = Path(__file__).resolve().parents[3]
PARAMS = load_parameters(ROOT / "configs" / "cad" / "xsens_dot2_watch.json")


def assert_close(actual, expected, tolerance=1e-6):
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{actual!r} != {expected!r}")


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


def test_cradle_and_coupon():
    cradle = build_cradle(PARAMS)
    coupon = build_fit_coupon(PARAMS)
    for name, shape in (("cradle", cradle), ("coupon", coupon)):
        if not shape.isValid() or len(shape.Solids) != 1 or shape.Volume <= 0:
            raise AssertionError(f"{name} must be one valid solid")
    assert_close(cradle.BoundBox.XLength, 60.8)
    assert_close(cradle.BoundBox.YLength, PARAMS.cradle_width_mm)
    if coupon.Volume >= cradle.Volume:
        raise AssertionError("fit coupon must be smaller than cradle")


def test_straps_and_positions():
    long_strap = build_long_strap(PARAMS)
    short_strap = build_short_strap(PARAMS)
    for name, shape in (("long_strap", long_strap), ("short_strap", short_strap)):
        if not shape.isValid() or len(shape.Solids) != 1 or shape.Volume <= 0:
            raise AssertionError(f"{name} must be one valid solid")
        hole_probe = Part.makeCylinder(
            (PARAMS.pin_diameter_mm + PARAMS.pin_hole_clearance_mm) * 0.45,
            PARAMS.strap_width_mm + 2.0,
            Vector(0.0, -PARAMS.strap_width_mm / 2.0 - 1.0, 2.8),
            Vector(0.0, 1.0, 0.0),
        )
        if shape.common(hole_probe).Volume > 1e-6:
            raise AssertionError(f"{name} hinge hole must remain open")
    positions = wrist_positions_mm(PARAMS)
    if len(positions) != 19:
        raise AssertionError("expected 19 ratchet positions")
    assert_close(positions[0], 140.0)
    assert_close(positions[-1], 212.0)
    for left, right in zip(positions, positions[1:]):
        assert_close(right - left, 4.0)
    assert_close(short_strap.BoundBox.XMax, PARAMS.short_strap_length_mm + 8.0)


def test_ratchet_components():
    components = {
        "housing": build_ratchet_housing(PARAMS),
        "pawl": build_pawl(PARAMS),
        "pin": build_pin(PARAMS),
    }
    for name, shape in components.items():
        if not shape.isValid() or len(shape.Solids) != 1 or shape.Volume <= 0:
            raise AssertionError(f"{name} must be one valid solid")
    assert_close(components["housing"].BoundBox.XLength, PARAMS.ratchet_housing_length_mm)
    assert_close(components["housing"].BoundBox.YLength, PARAMS.ratchet_housing_width_mm)
    assert_close(components["pin"].BoundBox.YLength, PARAMS.ratchet_housing_width_mm)
    for y0 in (9.55, -11.05):
        side_channel_probe = Part.makeBox(6.0, 1.35, 2.2, Vector(0.1, y0, 0.7))
        if components["housing"].common(side_channel_probe).Volume > 1e-6:
            raise AssertionError("ratchet housing side-arm channel must remain open")


def test_assembly_and_exports():
    components = build_components(PARAMS)
    by_name = {item.name: item for item in components}
    lug_x = PARAMS.cradle_length_mm / 2.0 + 4.0
    assert_close(by_name["Polymer_Pin_PETG"].translation_mm[0], -lug_x)
    assert_close(by_name["Polymer_Pin_PETG_2"].translation_mm[0], lug_x)
    assert_close(by_name["Polymer_Pin_PETG"].translation_mm[2], 2.8)
    assert_close(by_name["Polymer_Pin_PETG_2"].translation_mm[2], 2.8)
    validate_components(components)
    validate_sensor_fit(components)
    short_strap = by_name["Strap_Short_TPU"].shape.copy()
    short_strap.translate(Vector(*by_name["Strap_Short_TPU"].translation_mm))
    housing = by_name["Ratchet_Housing_PETG"].shape.copy()
    housing.translate(Vector(*by_name["Ratchet_Housing_PETG"].translation_mm))
    if short_strap.common(housing).Volume > 1e-5:
        raise AssertionError("short strap snap arms must fit the ratchet housing channels")
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
                raise AssertionError(f"missing export {path}")
        if len(result["component_steps"]) != 8:
            raise AssertionError("expected eight unique component STEP files")
        if len(result["stl_files"]) != 7:
            raise AssertionError("expected seven unique printable STL files")


test_rounded_box()
test_sensor_dummy()
test_cradle_and_coupon()
test_straps_and_positions()
test_ratchet_components()
test_assembly_and_exports()
print("FREECAD_TEST_PASS primitives sensor")
