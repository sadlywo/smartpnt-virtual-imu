from pathlib import Path

import Part
from FreeCAD import Vector
from xsens_dot2_watch.config import load_parameters
from xsens_dot2_watch.cradle import build_cradle, build_fit_coupon
from xsens_dot2_watch.geometry import rounded_box
from xsens_dot2_watch.ratchet import build_pawl, build_pin, build_ratchet_housing
from xsens_dot2_watch.sensor import build_sensor_dummy
from xsens_dot2_watch.straps import build_long_strap, build_short_strap, wrist_positions_mm

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


test_rounded_box()
test_sensor_dummy()
test_cradle_and_coupon()
test_straps_and_positions()
test_ratchet_components()
print("FREECAD_TEST_PASS primitives sensor")
