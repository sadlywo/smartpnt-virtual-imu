import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def frame():
    values = []
    for dataset, seq, count in [("alimu", "a", 20), ("penncosyvio", "b", 12)]:
        for t in range(count):
            r = dict(dataset_id=dataset, sequence_id=seq, original_sequence_id=seq,
                     pair_id=seq, low_sensor=seq, high_sensor="ref", timestamp_s=t / 40)
            for i, axis in enumerate(["gx", "gy", "gz", "ax", "ay", "az"]):
                r[axis + "_low"] = t + i
                r[axis + "_high"] = 2 * t + i + .1
            values.append(r)
    return pd.DataFrame(values)


def test_context_lengths_supervise_the_same_rows_and_values():
    from virtual_imu.ablations.data import CommonWindows
    d = CommonWindows(frame(), max_window=8, block_size=2, stride=2)
    for window in [2, 4, 8]:
        x, y = d.arrays(window)
        assert x.shape == (10, 6, window)
        assert y.shape == (10, 6, 2)
        np.testing.assert_array_equal(y, d.arrays(8)[1])
        center = (window - 2) // 2
        np.testing.assert_array_equal(x[..., center:center+2], d.arrays(2)[0])
    assert d.target_frame.sequence_id.tolist() == ["a"] * 14 + ["b"] * 6
    assert not d.target_frame.duplicated(["sequence_id", "timestamp_s"]).any()


def test_validation_normalizes_with_frozen_training_statistics():
    from virtual_imu.ablations.data import CommonWindows
    tr = CommonWindows(frame(), max_window=8, block_size=2, stride=2)
    other = frame()
    for col in [c for c in other if c.endswith("_low") or c.endswith("_high")]:
        other[col] += 100
    val = CommonWindows(other, max_window=8, block_size=2, stride=2, stats=tr.stats)
    assert val.stats == tr.stats
    assert val.arrays(8)[0].mean() > 10
    np.testing.assert_allclose(val.arrays(8)[1], tr.arrays(8)[1], atol=1e-5)


def test_balanced_orders_are_seeded_and_independent_of_context():
    from virtual_imu.ablations.data import CommonWindows, balanced_order
    d = CommonWindows(frame(), max_window=8, block_size=2, stride=2)
    np.testing.assert_array_equal(balanced_order(d, 7, 1), balanced_order(d, 7, 1))
    assert not np.array_equal(balanced_order(d, 7, 1), balanced_order(d, 7, 2))
    mass = pd.DataFrame({"dataset": d.anchor_datasets, "weight": d.weights}).groupby("dataset").weight.sum()
    np.testing.assert_allclose(mass.values, np.ones(2))


def test_physical_metrics_pool_samples_before_rmse_and_preserve_units():
    from virtual_imu.ablations.data import physical_metrics, grouped_score
    df = frame().iloc[:4].copy()
    high = df[[a + "_high" for a in ["gx", "gy", "gz", "ax", "ay", "az"]]].to_numpy()
    pred = high + np.array([1, 1, 1, 2, 2, 2])
    rows = physical_metrics(df, pred)
    overall = next(r for r in rows if r["aggregation"] == "overall")
    assert overall["gyro_rmse"] == pytest.approx(1)
    assert overall["accel_rmse"] == pytest.approx(2)
    assert overall["gyro_vector_rmse"] == pytest.approx(3 ** .5)
    assert grouped_score(df, high) == pytest.approx(0)


def test_invalid_or_cross_sequence_layout_is_rejected():
    from virtual_imu.ablations.data import CommonWindows
    with pytest.raises(ValueError):
        CommonWindows(frame(), max_window=8, block_size=3, stride=2)
    d = CommonWindows(frame(), max_window=8, block_size=2, stride=2)
    with pytest.raises(ValueError):
        d.arrays(5)
