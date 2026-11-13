"""Selection and source-support protections for the baseline supplement."""
import numpy as np
import pandas as pd
import pytest

from virtual_imu.context_study.classical_pipeline import (
    assert_support, choose_candidates, require_test_lock,
)


def test_selection_only_uses_valid_finite_validation_trials():
    trials = [dict(candidate_id='a', role='validation', primary=True, valid=True, score=.5),
              dict(candidate_id='b', role='validation', primary=False, valid=True, score=.3),
              dict(candidate_id='c', role='validation', primary=True, valid=False, score=None)]
    assert choose_candidates(trials) == {'primary3': 'a', 'expanded12': 'b'}
    trials[0]['role'] = 'test'
    with pytest.raises(ValueError, match='VALID'):
        choose_candidates(trials)


def test_test_payload_is_blocked_until_selection_lock_exists(tmp_path):
    with pytest.raises(RuntimeError, match='lock'):
        require_test_lock(tmp_path)
    (tmp_path/'selection.lock.json').write_text('{}', encoding='utf-8')
    require_test_lock(tmp_path)


def test_support_check_detects_reordered_or_changed_source_targets():
    frame = pd.DataFrame(dict(dataset_id=['alimu']*2, pair_id=['p']*2,
                              sequence_id=['s']*2, timestamp_s=[1., 2.]))
    from virtual_imu.ablations.data import support_digest
    receipt = dict(support_sha256=support_digest(frame), samples=2, anchors=1)
    assert_support(frame, receipt, anchors=1)
    with pytest.raises(ValueError, match='support'):
        assert_support(frame.iloc[::-1], receipt, anchors=1)
    with pytest.raises(ValueError, match='support'):
        assert_support(frame, receipt, anchors=2)


def test_arx_teacher_context_uses_no_extra_high_labels_outside_supervised_support():
    from virtual_imu.ablations.data import CommonWindows, LOW, HIGH
    from virtual_imu.context_study.classical_pipeline import _arrays
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(dict(dataset_id=['alimu']*400, pair_id=['p']*400,
                              sequence_id=['s']*400, timestamp_s=np.arange(400)/40))
    for name in LOW+HIGH:
        frame[name] = rng.normal(size=400)
    data = CommonWindows(frame)
    _, context = _arrays(data, 320, teacher_training=True)
    np.testing.assert_array_equal(context[0, :140], 0.)
    np.testing.assert_array_equal(context[0, 140:180], data.y[140:180])
    np.testing.assert_array_equal(context[1, 139], data.y[179])


def test_reference_replay_matches_arx_and_delta_without_using_feature_builder():
    from virtual_imu.context_study.classical_audit import reference_predict
    from virtual_imu.context_study.classical_models import predict
    rng = np.random.default_rng(17)
    x = rng.normal(size=(3, 20, 2))
    for delta in [False, True]:
        state = dict(offsets=np.arange(-3, 3), weights=rng.normal(size=(12+2*delta, 2))*.02,
                     bias=rng.normal(size=2)*.02, delta=delta,
                     ar_weights=rng.normal(size=(2, 2, 2))*.03)
        expected = reference_predict(state, x, center=8, block_size=4)
        np.testing.assert_allclose(predict(state, x, center=8, block_size=4), expected, atol=1e-12)
