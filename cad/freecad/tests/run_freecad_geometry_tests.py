from pathlib import Path

from xsens_dot2_watch.config import load_parameters
from xsens_dot2_watch.cradle import build_cradle, build_fit_coupon
from xsens_dot2_watch.geometry import rounded_box
from xsens_dot2_watch.sensor import build_sensor_dummy

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


test_rounded_box()
test_sensor_dummy()
test_cradle_and_coupon()
print("FREECAD_TEST_PASS primitives sensor")
