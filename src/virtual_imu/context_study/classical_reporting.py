"""Paper tables for genuinely matched information/support comparisons."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {'fir': 'FIR', 'arx_style': 'ARX-style (LOW + delta)',
          'arx_recursive': 'Recursive residual ARX'}


def _table(frame, columns):
    lines = ['| '+' | '.join(columns)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for _, row in frame.iterrows():
        lines.append('| '+' | '.join(format(row[c], '.7g') if isinstance(row[c], (float, np.floating))
                                      else str(row[c]) for c in columns)+' |')
    return '\n'.join(lines)


def neural_reference_rows(source_run):
    source = pd.read_csv(source_run/'overall_per_seed.csv')
    source = source[source.role == 'test']
    rows = []
    for _, row in source.iterrows():
        if row.stage == 'baseline':
            condition = 'shared' if row.held_sensor == 'shared' else row.held_sensor
            protocol, future = 'deterministic_source_baseline', '0'
        elif row.stage == 'window':
            condition, protocol = 'shared', 'window80'
            future = 'full' if row.family in ['cnn_bigru', 'cnn_bilstm32', 'cnn_transformer'] else '21' if row.family == 'cnn_tcn' else '6'
        elif row.stage == 'search':
            condition, protocol = 'shared', 'search40'
            future = 'full' if row.family in ['cnn_bigru', 'cnn_transformer', 'transformer', 'informer'] else '21' if row.family == 'cnn_tcn' else '6'
        elif row.stage == 'future':
            condition, protocol, future = 'shared', 'controlled40', str(row.lookahead)
        elif row.stage == 'mechanism':
            if row.condition != 'd3_k5_w64_none_identity':
                continue
            condition, protocol, future = 'shared', 'causal_frontend40', '0'
        elif row.stage == 'generalization':
            condition, protocol = row.held_sensor, 'held_device40'
            future = '6' if row.family == 'cnn_gru' else 'full'
        else:
            continue
        rows.append(dict(held_sensor=condition, window=int(row.window), lookahead=future,
                         protocol=protocol, family=row.family, seed=int(row.seed), score=row.score,
                         gyro_rmse=row.gyro_rmse, accel_rmse=row.accel_rmse))
    return pd.DataFrame(rows)


def build_report(run, source_run, cfg):
    run, source_run = Path(run), Path(source_run)
    overall = pd.read_csv(run/'overall.csv')
    test = overall[overall.role == 'test'].copy()
    refs = neural_reference_rows(source_run)
    refs.to_csv(run/'neural_references.csv', index=False)
    matched = []
    for _, baseline in test.iterrows():
        exact = refs[(refs.held_sensor == baseline.held_sensor) & (refs.window == baseline.window) &
                     (refs.lookahead.astype(str) == str(baseline.lookahead)) &
                     (~refs.protocol.eq('deterministic_source_baseline'))]
        for (protocol, family), group in exact.groupby(['protocol', 'family']):
            matched.append(dict(held_sensor=baseline.held_sensor, window=baseline.window,
                                lookahead=str(baseline.lookahead), tier=baseline.tier,
                                classical=baseline.family, neural=family, neural_protocol=protocol,
                                classical_S=baseline.score, neural_S=float(group.score.mean()),
                                neural_seed_SD=float(group.score.std(ddof=1)),
                                classical_change_vs_neural_pct=100*(baseline.score/group.score.mean()-1)))
    pd.DataFrame(matched).to_csv(run/'matched_comparisons.csv', index=False)
    kept = ['held_sensor', 'window', 'lookahead', 'tier', 'family', 'score', 'gyro_rmse', 'accel_rmse', 'candidate']
    test[kept].to_csv(run/'test_summary.csv', index=False)
    text = [
        '# Matched ALiMU FIR / ARX Baselines', '',
        'All results use the sealed context-study center40 targets, recording splits and TRAIN-only normalization.',
        'Source run: `'+str(source_run)+'`. This supplement trains no neural models and leaves the source read-only.', '',
        '## Methods and Information Bounds', '',
        '- FIR is multivariate LOW lag regression on normalized residual targets.',
        '- ARX-style retains the historical LOW lag plus first-difference definition; it has no output autoregression.',
        '- Recursive residual ARX includes past residual outputs during TRAIN identification and uses only its own predicted residuals during VALID/TEST. Each input anchor starts from zero normalized residual state.',
        '- Each lag is bounded by its own input anchor. Future0/future6 use no later LOW than t/t+6; full uses only its supplied window.',
        '- Only TRAIN center40 responses enter least squares. ARX teacher lags also use only rows in the union of these supervised TRAIN targets; other residual history is zero. HIGH is never a prediction argument.',
        '- Primary3 has three declared history/alpha packages per family. Expanded12 is a separate 3x4 sensitivity search. Both select on VALID and lock before new TEST access.',
        '- Stable ARX candidates require the TRAIN-fitted companion spectral radius below the configured bound. Rejected trials remain in trials.csv.',
        '- Deterministic baselines have one fit; no duplicated seed error bars. Neural SD describes three optimization seeds.',
        '- Input availability matches the declared window/future bound; finite-order models can select a shorter span within it.',
        '- Historical TEST had already been examined. This is a retrospective comparator supplement.', '',
        '## Shared Model: Primary Three-Candidate Results', '',
        _table(test[(test.held_sensor == 'shared') & (test.tier == 'primary3')],
               ['window', 'lookahead', 'family', 'score', 'gyro_rmse', 'accel_rmse']), '',
        '## Held Device: Primary Three-Candidate Results', '',
        _table(test[(test.held_sensor != 'shared') & (test.tier == 'primary3')],
               ['held_sensor', 'lookahead', 'family', 'score', 'gyro_rmse', 'accel_rmse']), '',
        '## Expanded Search Sensitivity', '',
        _table(test[test.tier == 'expanded12'],
               ['held_sensor', 'window', 'lookahead', 'family', 'score', 'candidate']), '',
        '## Reading the Neural Comparison', '',
        'matched_comparisons.csv pairs only equal cohort, window and future bounds. window80, search40 and controlled40 remain separately labeled; their budgets and architectures are not pooled into one ranking.',
        'Source CNN-TCN window/search models have a21-point future dependency; this supplement does not treat future6/full as an exact match for them.',
        'Raw LOW and the source affine controls are retained in neural_references.csv for reference.', '',
        '## Traceability', '',
        'config.yaml, data_manifest.json, provenance.json, analysis_plan.json and selection.lock.json identify the experiment. states/ retains every candidate. predictions/ and supports/ retain each selected model output and exact target identities. metrics.csv gives overall/device/recording/axis results; numerical_audit.json gives independent checks. artifact_manifest.json and complete.json seal the run.', '',
        'Known preprocessing delay is25ms. Model future6 adds150ms; these bounds do not include transport or computation.', '',
        '![Matched shared results](figures/shared_classical.png)', '',
        '![Held-device results](figures/held_device_classical.png)', '',
    ]
    (run/'report.md').write_text('\n'.join(text), encoding='utf-8')
    from virtual_imu.viz.plots_context_classical import make_figures
    make_figures(run, test, refs)
    summary = {}
    for tier in ['primary3', 'expanded12']:
        part = test[(test.tier == tier) & (test.held_sensor == 'shared') & (test.window == 320)]
        summary[tier] = [{k: row[k] for k in ['lookahead', 'family', 'score', 'gyro_rmse', 'accel_rmse']}
                         for _, row in part.iterrows()]
    (run/'answers.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
