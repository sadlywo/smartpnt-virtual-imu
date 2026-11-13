"""Matched classical supplement; source runs remain read-only."""
from __future__ import annotations

import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from virtual_imu.ablations.data import CommonWindows, HIGH, LOW, grouped_score, support_digest
from virtual_imu.context_study.classical_models import (
    add_delta_statistics, fit_ridge, load_state, make_state, predict,
    save_state, spectral_radius, sufficient_statistics,
)
from virtual_imu.context_study.data import StudySource
from virtual_imu.fir_arx_research.data import sha256_file

LOG = logging.getLogger(__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n',
                          encoding='utf-8')
    LOG.info('WRITE %s', path)


def assert_support(frame, receipt, *, anchors):
    if (len(frame) != receipt['samples'] or anchors != receipt['anchors'] or
            support_digest(frame) != receipt['support_sha256']):
        raise ValueError('Source support mismatch')


def choose_candidates(trials):
    if any(row['role'] != 'validation' for row in trials):
        raise ValueError('Selection must use VALID only')
    result = {}
    for tier in ['primary3', 'expanded12']:
        candidates = [r for r in trials if r['valid'] and (tier == 'expanded12' or r['primary'])]
        if not candidates or any(not np.isfinite(r['score']) for r in candidates):
            raise ValueError('No finite valid candidate in '+tier)
        result[tier] = min(candidates, key=lambda r: (r['score'], r['candidate_id']))['candidate_id']
    return result


def require_test_lock(run):
    if not (Path(run)/'selection.lock.json').exists():
        raise RuntimeError('TEST payload requires a selection lock')


def groups_for_config(cfg):
    groups = [dict(held='shared', window=int(w), future='full') for w in cfg['shared_full_windows']]
    groups += [dict(held='shared', window=320, future=int(f)) for f in cfg['shared_lookaheads']]
    groups += [dict(held=s, window=320, future=f) for s in cfg['held_sensors']
               for f in cfg.get('held_lookaheads', ['full'])]
    for group in groups:
        group['id'] = '%s__w%d__future%s' % (group['held'], group['window'], group['future'])
    return groups


def package_offsets(group, package, block_size=40):
    center = (group['window']-block_size)//2
    max_distance = center+block_size-1
    if group['future'] == 'full':
        radius = min(package['full_radius'], max_distance)
        return np.arange(-radius, radius+1)
    past = min(package['past_points']-1, max_distance)
    return np.arange(-past, int(group['future'])+1)


def independent_metrics(frame, prediction):
    """Separate NumPy scorer: no production scoring function is called."""
    high = frame[HIGH].to_numpy(float)
    raw = frame[LOW].to_numpy(float)
    error = prediction-high
    scores = []
    for ix in frame.groupby('pair_id', sort=False).indices.values():
        ratios = []
        for part in [slice(0, 3), slice(3, 6)]:
            ratios.append(np.sqrt(np.sum(error[ix, part]**2)/error[ix, part].size) /
                          np.sqrt(np.sum((raw[ix, part]-high[ix, part])**2)/error[ix, part].size))
        scores.append(sum(ratios)/2)
    return dict(score=float(np.mean(scores)), gyro_rmse=float(np.sqrt(np.mean(error[:, :3]**2))),
                accel_rmse=float(np.sqrt(np.mean(error[:, 3:]**2))))


def _source_data(root, source_run, data_cfg, held, role, stats, run):
    if role == 'test':
        require_test_lock(run)
    LOG.info('READ original context cohort=%s role=%s; SKIP unrequested payloads', held, role)
    source = StudySource(root, data_cfg, held_sensor=None if held == 'shared' else held)
    data = CommonWindows(source.load(role), max_window=320, block_size=40, stride=40, stats=stats)
    receipt = json.loads((source_run/'supports'/(held+'__'+role+'.json')).read_text(encoding='utf-8'))
    assert_support(data.target_frame, receipt, anchors=len(data))
    for key in data.stats:
        np.testing.assert_allclose(data.stats[key], receipt['stats'][key], rtol=0, atol=0)
    return data


def _arrays(data, window, teacher_training=False):
    x, _ = data.arrays(window)
    offset = (320-window)//2
    y = np.stack([data.y[s+offset:s+offset+window] for s in data.starts])
    if teacher_training:
        eligible = np.zeros(len(data.frame), dtype=bool)
        eligible[data.target_indices] = True
        allowed = np.stack([eligible[s+offset:s+offset+window] for s in data.starts])
        y = np.where(allowed[:, :, None], y, 0.)
    return x.transpose(0, 2, 1), y


def _physical(data, normalized):
    return data.physical(normalized.transpose(0, 2, 1))


def _metric_rows(data, prediction, identity, role):
    from virtual_imu.context_study.evaluation import metric_rows
    rows = metric_rows(data, prediction, identity, role)
    return rows


def _fit_group(group, train, valid, cfg, run):
    LOG.info('FIT group %s; READ TRAIN/VALID; SKIP TEST', group['id'])
    x, y = _arrays(train, group['window'], teacher_training=True)
    vx, _ = _arrays(valid, group['window'])
    center = (group['window']-40)//2
    offsets = package_offsets(group, cfg['search']['packages'][-1])
    order = max(p['ar_order'] for p in cfg['search']['packages'])
    started = time.perf_counter()
    stats = sufficient_statistics(x, y, center=center, block_size=40, offsets=offsets,
                                  max_ar_order=order,
                                  batch_anchors=cfg['execution']['gram_batch_anchors'],
                                  device=cfg['execution']['device'])
    with_delta = add_delta_statistics(stats, offsets, 6)
    offset_indices = {int(o): np.arange(i*6, i*6+6) for i, o in enumerate(offsets)}
    selected = {}
    all_trials = []
    for method in cfg['methods']:
        trials = []
        for package in cfg['search']['packages']:
            selected_offsets = package_offsets(group, package)
            indices = np.concatenate([offset_indices[int(o)] for o in selected_offsets])
            ar_order = package['ar_order'] if method == 'arx_recursive' else 0
            if ar_order:
                indices = np.r_[indices, np.arange(len(offsets)*6, len(offsets)*6+ar_order*6)]
            if method == 'arx_style':
                indices = np.r_[indices, np.arange(len(stats['sum_x']), len(stats['sum_x'])+6)]
            fitted = fit_ridge(with_delta if method == 'arx_style' else stats,
                               indices, cfg['search']['alphas'])
            for alpha, (weights, bias) in fitted.items():
                candidate = package['id']+'__a'+format(alpha, '.8g')
                state = make_state(weights, bias, selected_offsets, 6,
                                   delta=method == 'arx_style', ar_order=ar_order)
                radius = spectral_radius(state['ar_weights'])
                row = dict(group=group['id'], method=method, candidate_id=candidate,
                           role='validation', primary=alpha == package['primary_alpha'],
                           package=package['id'], alpha=alpha, ar_order=ar_order,
                           past_points=int(-selected_offsets[0]+1),
                           future_points=int(selected_offsets[-1]), spectral_radius=radius,
                           valid=radius < cfg['search']['max_ar_spectral_radius'], score=None,
                           parameters=int(weights.size+bias.size))
                state_path = run/'states'/(group['id']+'__'+method+'__'+candidate+'.npz')
                save_state(state_path, state)
                LOG.info('WRITE candidate state %s', state_path)
                row['state'] = state_path.relative_to(run).as_posix()
                row['state_sha256'] = sha256_file(state_path)
                if row['valid']:
                    try:
                        normal = predict(state, vx, center=center, block_size=40,
                                         batch_anchors=cfg['execution']['prediction_batch_anchors'])
                        row['score'] = grouped_score(valid.target_frame, _physical(valid, normal))
                    except (ValueError, FloatingPointError) as exc:
                        row.update(valid=False, reason=str(exc))
                else:
                    row['reason'] = 'TRAIN-fitted AR companion spectral radius exceeds declared bound'
                trials.append(row)
                LOG.info('VALID %s %s %s score=%s valid=%s', group['id'], method,
                         candidate, row['score'], row['valid'])
        choice = choose_candidates(trials)
        for tier, candidate in choice.items():
            row = next(r for r in trials if r['candidate_id'] == candidate)
            selected[group['id']+'__'+method+'__'+tier] = dict(row, tier=tier, condition=group)
        all_trials += trials
    LOG.info('FIT complete %s elapsed=%.1fs', group['id'], time.perf_counter()-started)
    return selected, all_trials


def _freeze(root, run, source_run, cfg, config_path):
    run.mkdir(parents=True, exist_ok=False)
    (run/'states').mkdir()
    (run/'predictions').mkdir()
    (run/'supports').mkdir()
    handler = logging.FileHandler(run/'pipeline.log', encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(handler)
    (run/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    shutil.copy2(source_run/'data_manifest.json', run/'data_manifest.json')
    paths = sorted((root/'src/virtual_imu').rglob('*.py'))
    paths += [root/'scripts/run_context_baselines.py', config_path]
    sources = {}
    for path in paths:
        relative = path.relative_to(root).as_posix()
        sources[relative] = sha256_file(path)
        dest = run/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    source_manifest = json.loads((source_run/'artifact_manifest.json').read_text(encoding='utf-8'))['files']
    inputs = ['config.yaml', 'data_manifest.json', 'complete.json', 'overall_per_seed.csv',
              'summary.csv', 'fit_registry.csv']
    inputs += [p.relative_to(source_run).as_posix() for p in (source_run/'supports').glob('*.json')]
    original = {}
    for relative in inputs:
        digest = sha256_file(source_run/relative)
        if relative in source_manifest and digest != source_manifest[relative]:
            raise ValueError('Sealed source hash mismatch '+relative)
        original[relative] = digest
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True, check=True)
    dirty = subprocess.run(['git', 'status', '--short'], cwd=root, capture_output=True, text=True, check=True)
    write_json(run/'provenance.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
               source_run=str(source_run), source_inputs=original, source_sha256=sources,
               config_sha256=sha256_file(run/'config.yaml'), git_commit=git.stdout.strip(),
               working_tree_status=dirty.stdout, python=sys.version, executable=sys.executable,
               platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
               cuda_gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               retrospective_test_comparison=True, newly_trained_neural_models=0))
    LOG.info('WRITE config, data manifest and %d source snapshots; SKIP source-run writes', len(sources))


def verify(run):
    run = Path(run)
    manifest = json.loads((run/'artifact_manifest.json').read_text(encoding='utf-8'))['files']
    for relative, digest in manifest.items():
        path = (run/relative).resolve()
        if run.resolve() not in path.parents or sha256_file(path) != digest:
            raise ValueError('Artifact hash mismatch '+relative)
    provenance = json.loads((run/'provenance.json').read_text(encoding='utf-8'))
    source_run = Path(provenance['source_run'])
    for relative, digest in provenance['source_inputs'].items():
        if sha256_file(source_run/relative) != digest:
            raise ValueError('Source run changed '+relative)
    LOG.info('READ sealed artifacts and source identities; SKIP writes; PASS %d hashes', len(manifest))
    return dict(status='PASS', artifact_hashes=len(manifest), source_inputs=len(provenance['source_inputs']))


def execute(root, config_path):
    root = Path(root).resolve()
    config_path = (root/config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    source_run = (root/cfg['source_run']).resolve()
    if not (source_run/'complete.json').exists():
        raise ValueError('Source study is not complete')
    run = root/cfg['output']['root']/cfg['experiment_id']
    if run.exists():
        raise FileExistsError('Use a fresh experiment_id; existing run preserved: '+str(run))
    _freeze(root, run, source_run, cfg, config_path)
    torch.set_num_threads(cfg['execution']['threads'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    source_cfg = yaml.safe_load((source_run/'config.yaml').read_text(encoding='utf-8'))
    groups = groups_for_config(cfg)
    write_json(run/'analysis_plan.json', dict(groups=groups, methods=cfg['methods'],
               primary_trials_per_family=3, expanded_trials_per_family=12,
               selection_metric='source study VALID S', fit_targets='same TRAIN center40 only',
               input='LOW6 only; anchor-local normalized zero padding',
               recursive_arx='Only supervised TRAIN rows supply output lags; predicted residual feedback only at VALID/TEST',
               reset='every input anchor', random_seed=cfg['seed'], test_loaded=False))
    selections, trials = {}, []
    for held in ['shared']+cfg['held_sensors']:
        train = _source_data(root, source_run, source_cfg['data'], held, 'train', None, run)
        valid = _source_data(root, source_run, source_cfg['data'], held, 'validation', train.stats, run)
        write_json(run/'supports'/(held+'__stats.json'), train.stats)
        for role, data in [('train', train), ('validation', valid)]:
            shutil.copy2(source_run/'supports'/(held+'__'+role+'.json'),
                         run/'supports'/(held+'__'+role+'.json'))
        for group in [g for g in groups if g['held'] == held]:
            chosen, rows = _fit_group(group, train, valid, cfg, run)
            selections.update(chosen)
            trials += rows
            pd.DataFrame(trials).to_csv(run/'trials.csv', index=False)
            write_json(run/'progress.json', dict(completed_groups=len(selections)//6,
                       total_groups=len(groups), completed_trials=len(trials), test_loaded=False))
        del train, valid
    write_json(run/'selection.lock.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
               source_roles_loaded=['train', 'validation'], test_loaded=False,
               config_sha256=sha256_file(run/'config.yaml'), selections=selections))
    _evaluate(root, source_run, source_cfg, cfg, run, selections)
    return run


def _evaluate(root, source_run, source_cfg, cfg, run, selections):
    require_test_lock(run)
    write_json(run/'test_access.json', dict(selection_lock_sha256=sha256_file(run/'selection.lock.json'),
               created_utc=datetime.now(timezone.utc).isoformat(), reason='all VALID selections locked'))
    all_rows, audit_rows, causality = [], [], []
    for held in ['shared']+cfg['held_sensors']:
        stats = json.loads((run/'supports'/(held+'__stats.json')).read_text(encoding='utf-8'))
        for role in ['validation', 'test']:
            data = _source_data(root, source_run, source_cfg['data'], held, role, stats, run)
            data.target_frame.to_parquet(run/'supports'/(held+'__'+role+'.parquet'), index=False)
            shutil.copy2(source_run/'supports'/(held+'__'+role+'.json'),
                         run/'supports'/(held+'__'+role+'.json'))
            arrays = {}
            for identity, row in selections.items():
                group = row['condition']
                if group['held'] != held:
                    continue
                if sha256_file(run/row['state']) != row['state_sha256']:
                    raise ValueError('Selected model state changed')
                if group['window'] not in arrays:
                    arrays[group['window']] = _arrays(data, group['window'])[0]
                x = arrays[group['window']]
                center = (group['window']-40)//2
                state = load_state(run/row['state'])
                normal = predict(state, x, center=center, block_size=40,
                                 batch_anchors=cfg['execution']['prediction_batch_anchors'])
                physical = _physical(data, normal)
                receipt = dict(job_id=identity, stage='classical', family=row['method'],
                               condition=group['id'], window=group['window'], seed=0,
                               candidate=row['candidate_id'], held_sensor=held,
                               lookahead=str(group['future']), tier=row['tier'],
                               depth=0, norm='train_standardization', activation='identity',
                               kernel=len(state['offsets']), dilation=1, width=6)
                metrics = _metric_rows(data, physical, receipt, role)
                all_rows += metrics
                audit = independent_metrics(data.target_frame, physical)
                overall = next(m for m in metrics if m['aggregation'] == 'overall')
                error = max(abs(overall[k]-audit[k]) for k in audit)
                if error > 1e-11:
                    raise ValueError('Independent metric recomputation failed')
                if role == 'validation' and abs(audit['score']-row['score']) > 1e-11:
                    raise ValueError('VALID selection replay failed')
                audit_rows.append(dict(job_id=identity, role=role, maximum_error=error))
                np.savez_compressed(run/'predictions'/(identity+'__'+role+'.npz'),
                                    prediction=physical, support_sha256=data.support_sha256)
                if role == 'validation':
                    from virtual_imu.context_study.classical_audit import reference_predict
                    independent = reference_predict(state, x[:2], center=center, block_size=40)
                    replay_error = float(np.max(np.abs(independent-normal[:2])))
                    if replay_error > 1e-8:
                        raise ValueError('Independent state replay failed')
                    audit_rows[-1]['independent_state_replay_error'] = replay_error
                if role == 'validation' and group['future'] != 'full':
                    probe = x[:2].copy()
                    t, future = center+10, int(group['future'])
                    changed = probe.copy()
                    changed[:, t+future+1:] += 100
                    a = predict(state, probe, center=center, block_size=40)
                    b = predict(state, changed, center=center, block_size=40)
                    delta = float(np.max(np.abs(a[:, 10]-b[:, 10])))
                    if delta > 1e-10:
                        raise ValueError('Future information bound violated')
                    causality.append(dict(job_id=identity, future=future, perturbation_error=delta))
                LOG.info('EVALUATE %s %s S=%.6f; WRITE predictions and metrics', identity, role, audit['score'])
            del arrays, data
    pd.DataFrame(all_rows).to_csv(run/'metrics.csv', index=False)
    pd.DataFrame(all_rows).query("aggregation == 'overall'").to_csv(run/'overall.csv', index=False)
    write_json(run/'numerical_audit.json', dict(status='PASS', independent_metric_recomputations=audit_rows,
               future_checks=causality, max_metric_error=max(r['maximum_error'] for r in audit_rows),
               number_of_selected_models=len(selections), validation_only_selection=True))
    from virtual_imu.context_study.classical_reporting import build_report
    build_report(run, source_run, cfg)
    provenance = json.loads((run/'provenance.json').read_text(encoding='utf-8'))
    for relative, digest in provenance['source_inputs'].items():
        if sha256_file(source_run/relative) != digest:
            raise ValueError('Source run changed during supplement')
    for relative, digest in provenance['source_sha256'].items():
        if sha256_file(root/relative) != digest:
            raise ValueError('Implementation changed during supplement '+relative)
    manifest = {p.relative_to(run).as_posix(): sha256_file(p) for p in run.rglob('*')
                if p.is_file() and p.name not in ['pipeline.log', 'artifact_manifest.json', 'complete.json']}
    write_json(run/'artifact_manifest.json', dict(files=manifest, excluded_mutable_files=['pipeline.log']))
    verification = verify(run)
    write_json(run/'complete.json', dict(status='PASS', selected_models=len(selections),
               trials=len(pd.read_csv(run/'trials.csv')), verification=verification,
               artifact_manifest_sha256=sha256_file(run/'artifact_manifest.json')))
    LOG.info('COMPLETE %s; SKIP source-run writes and neural retraining', run)
