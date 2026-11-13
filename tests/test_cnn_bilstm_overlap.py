import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def module():
    path = ROOT / "scripts/evaluate_cnn_bilstm_overlap.py"
    assert path.is_file(), "Overlap diagnostic is not implemented"
    spec = importlib.util.spec_from_file_location("cnn_overlap_diagnostic", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def data():
    from virtual_imu.ablations.data import CommonWindows, LOW, HIGH
    frame = pd.DataFrame({c: np.zeros(12) for c in LOW + HIGH})
    frame = frame.assign(dataset_id="alimu", pair_id="p", sequence_id="s", original_sequence_id="s", timestamp_s=np.arange(12) / 40)
    return CommonWindows(frame, max_window=8, block_size=2, stride=2)


def test_overlap_is_averaged_before_metrics_on_identical_target_support():
    m, d = module(), data()
    prediction = np.stack([np.full((6, 8), v) for v in [0., 10., 20.]])
    output, counts = m.stitch_on_common_support(d, prediction)
    np.testing.assert_array_equal(output, np.repeat(np.array([5., 10., 10., 10., 10., 15.])[:, None], 6, axis=1))
    np.testing.assert_array_equal(counts, [2, 3, 3, 3, 3, 2])
    assert len(output) == len(d.target_frame)


def test_physical_inverse_uses_same_low_and_training_residual_statistics():
    m, d = module(), data()
    d.stats["residual_mean"] = [2.] * 6
    d.stats["residual_std"] = [3.] * 6
    output, _ = m.stitch_on_common_support(d, np.ones((3, 6, 8)))
    np.testing.assert_array_equal(output, np.full((6, 6), 5.))


def test_wrong_prediction_shape_is_rejected():
    m, d = module(), data()
    with pytest.raises(ValueError):
        m.stitch_on_common_support(d, np.zeros((2, 6, 8)))
