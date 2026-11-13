"""Matched classical baselines in the repository IMU paper style."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from virtual_imu.viz.style import apply_paper_style, save_figure

LABELS = {'fir': 'FIR', 'arx_style': 'ARX-style', 'arx_recursive': 'Recursive ARX'}
COLORS = ['#1f77b4', '#c95f3f', '#2a9d8f']


def make_figures(run, test, refs):
    run = Path(run)
    out = run/'figures'
    out.mkdir(exist_ok=True)
    apply_paper_style()
    plt.rcParams['svg.fonttype'] = 'none'
    primary = test[test.tier == 'primary3']
    files = []
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)
    for ax, future in zip(axes, ['0', '6', 'full']):
        part = primary[(primary.held_sensor == 'shared') & (primary.window == 320) &
                       (primary.lookahead.astype(str) == future)]
        for index, family in enumerate(LABELS):
            value = part.loc[part.family == family, 'score'].iloc[0]
            ax.bar(index, value, color=COLORS[index], width=.6, label=LABELS[family])
        neural = refs[(refs.held_sensor == 'shared') & (refs.window == 320) &
                      (refs.lookahead == future) & refs.protocol.isin(['controlled40', 'causal_frontend40', 'search40'])]
        if len(neural):
            best = neural.groupby(['protocol', 'family']).score.mean().sort_values()
            ax.axhline(best.iloc[0], color='#333333', ls='--', lw=1,
                       label=best.index[0][1]+' ('+best.index[0][0]+')')
        ax.axhline(1, color='#999999', lw=.8, ls=':')
        ax.set_title('Future '+future+(' (150 ms)' if future == '6' else ''))
        ax.set_xticks([])
        ax.set_xlabel('320-point input')
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.19), fontsize=5.7, frameon=False)
    axes[0].set_ylabel('Relative error score S (lower better)')
    fig.subplots_adjust(bottom=.35, wspace=.15)
    files += save_figure(fig, out, 'shared_classical')
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    sensors = ['BMX160', 'LSM9DS1', 'MPU9250']
    for ax, future in zip(axes, ['6', 'full']):
        part = primary[(primary.held_sensor != 'shared') & (primary.lookahead.astype(str) == future)]
        for index, family in enumerate(LABELS):
            values = [part[(part.held_sensor == s) & (part.family == family)].score.iloc[0] for s in sensors]
            ax.bar(np.arange(3)+(index-1)*.22, values, width=.21, color=COLORS[index], label=LABELS[family])
        ax.axhline(1, color='#555555', lw=1, ls='--', label='Raw LOW')
        ax.set_xticks(np.arange(3), sensors)
        ax.set_title('Future '+future)
    axes[0].set_ylabel('Relative error score S (lower better)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.subplots_adjust(bottom=.22, wspace=.15)
    files += save_figure(fig, out, 'held_device_classical')
    (out/'figure_index.md').write_text(
        '# Matched Classical Baseline Figures\n\n'
        '- shared_classical: Three-candidate VALID-selected deterministic FIR/ARX baselines on identical320-point anchors and center40 TEST targets. Panels separate0/6/full future bounds. Dashed lines identify the lowest neural TEST mean among the labeled source protocols under that bound; they are descriptive references, not new selections. Dotted line is Raw S=1.\n'
        '- held_device_classical: Device and recording isolation, source-only TRAIN/VALID fitting, separate6/full future bounds. Deterministic methods have no seed uncertainty bars.\n',
        encoding='utf-8')
    return files
