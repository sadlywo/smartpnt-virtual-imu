"""Information-bound and numerical controls for matched linear baselines."""
import numpy as np
import pytest

from virtual_imu.context_study.classical_models import (
    fit_ridge, lag_features, predict, sufficient_statistics,
)


def test_lags_zero_pad_each_anchor_without_crossing_its_boundary():
    x = np.array([[[1.], [2.], [3.]], [[100.], [200.], [300.]]])
    got = lag_features(x, np.array([0, 2]), np.array([-1, 0, 1]))
    np.testing.assert_array_equal(got, [[[0., 1., 2.], [2., 3., 0.]],
                                        [[0., 100., 200.], [200., 300., 0.]]])


@pytest.mark.parametrize('order', [0, 2])
@pytest.mark.parametrize('future', [0, 2])
def test_predictions_cannot_read_after_the_declared_future(order, future):
    rng = np.random.default_rng(4)
    offsets = np.arange(-5, future + 1)
    state = dict(offsets=offsets, weights=rng.normal(size=(len(offsets)*2, 2))*.01,
                 bias=np.zeros(2), ar_weights=np.zeros((order, 2, 2)), delta=False)
    if order:
        state['ar_weights'][0] = np.eye(2)*.2
    x = rng.normal(size=(3, 20, 2))
    altered = x.copy()
    altered[:, 8+future+1:] += 100
    a = predict(state, x, center=8, block_size=2)
    b = predict(state, altered, center=8, block_size=2)
    np.testing.assert_allclose(a[:, 0], b[:, 0], atol=1e-12, rtol=0)


def test_recursive_arx_uses_its_own_outputs_and_resets_every_anchor():
    state = dict(offsets=np.array([0]), weights=np.ones((1, 1)), bias=np.zeros(1),
                 ar_weights=np.array([[[.5]]]), delta=False)
    x = np.array([[[2.], [0.], [0.]], [[8.], [0.], [0.]]])
    got = predict(state, x, center=1, block_size=2)
    np.testing.assert_allclose(got[:, :, 0], [[1., .5], [4., 2.]])


def test_sufficient_statistics_fit_only_center_targets_and_match_direct_ridge():
    rng = np.random.default_rng(8)
    x, y = rng.normal(size=(9, 12, 2)), rng.normal(size=(9, 12, 2))
    offsets = np.array([-2, -1, 0, 1])
    stats = sufficient_statistics(x, y, center=4, block_size=3, offsets=offsets,
                                  max_ar_order=2, batch_anchors=2, device='cpu')
    design = lag_features(x, np.arange(4, 7), offsets)
    previous = np.concatenate([y[:, np.arange(4, 7)-k] for k in [1, 2]], axis=-1)
    joined = np.concatenate([design, previous], axis=-1).reshape(27, -1)
    target = y[:, 4:7].reshape(27, 2)
    indices = np.arange(joined.shape[1])
    weights, bias = fit_ridge(stats, indices, [.1])[.1]
    aug = np.column_stack([joined, np.ones(len(joined))])
    penalty = np.eye(aug.shape[1])*.1
    penalty[-1, -1] = 0
    expected = np.linalg.solve(aug.T@aug+penalty, aug.T@target)
    np.testing.assert_allclose(weights, expected[:-1], atol=1e-11)
    np.testing.assert_allclose(bias, expected[-1], atol=1e-11)
    changed = y.copy()
    changed[:, 9:] = 1e6
    again = sufficient_statistics(x, changed, center=4, block_size=3, offsets=offsets,
                                  max_ar_order=2, batch_anchors=3, device='cpu')
    np.testing.assert_allclose(stats['rhs'], again['rhs'], atol=1e-12)


def test_prediction_rejects_nonfinite_input():
    state = dict(offsets=np.array([0]), weights=np.ones((1, 1)), bias=np.zeros(1),
                 ar_weights=np.zeros((0, 1, 1)), delta=False)
    with pytest.raises(ValueError, match='finite'):
        predict(state, np.array([[[np.nan]]]), center=0, block_size=1)
