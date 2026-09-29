"""
Figures, primary features only: mean fold change with error bars set by ERRORBAR (linear axis, or log axis
when SCALE = 'log').
Main figure (between_group_drug_slice_bars): all groups per feature; * over a bar = one-sample test vs no
change (BH-FDR across the primary features), # bracket = one-way ANOVA with Holm-Sidak vs WT.
Supplementary ({group}_slice_bars): Baseline/Drug per group, y-axes shared across groups.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator

from slice_data import (slice_table, GROUPS, fig_label,
                        ALPHA, REFERENCE, SCALE, ERRORBAR)
from slice_robust import (PRIMARY_FEATURES, within_group, between_group, CI_LEVEL, stars,
                          MEAN_NAME, p_reported)

COLORS    = {'WT': '#4878CF', 'AV': '#6ACC65', 'IP': '#D65F5F'}
PLOTS_DIR = os.path.join(os.getcwd(), 'plots_slice')
os.makedirs(PLOTS_DIR, exist_ok=True)

WITHIN  = {c: within_group(c) for c in ['drug']}
BETWEEN = {c: between_group(c) for c in ['drug']}

LOG  = SCALE == 'log'
STEP = 1.30                       # bracket spacing on a log axis (multiplicative)
TEST = 't-test on log FC vs 0' if LOG else 'one-sample t-test vs 1'
ERR_LABEL = {'ci': f'{int(CI_LEVEL * 100)}% CI', 'sem': 'SEM', 'sd': 'SD'}[ERRORBAR]
SUFFIX    = '' if ERRORBAR == 'ci' else f'_{ERRORBAR}'   # SEM / SD figures get their own files
N_PRIMARY = len(PRIMARY_FEATURES)

# n = slices/animals per group (animal = data folder)
_ids = slice_table[['group', 'data_folder', 'slice_id']].drop_duplicates()
N_LABEL = {g: f'{(_ids["group"] == g).sum()}/{_ids.loc[_ids["group"] == g, "data_folder"].nunique()}'
           for g in GROUPS}


# --- Helpers ---

def _slice_fc(group, feature, condition):
    """Per-slice fold changes, for the dots."""
    m = ((slice_table['group'] == group) & (slice_table['feature'] == feature) &
         (slice_table['condition'] == condition))
    v = slice_table.loc[m, 'fc'].dropna().values
    return v[v > 0] if LOG else v


def _within_row(group, feature, condition):
    r = WITHIN[condition]
    r = r[(r['group'] == group) & (r['feature'] == feature)]
    return None if r.empty else r.iloc[0]


def _err(row):
    """Error-bar ends for a within-group row: the CI, or mean ± SEM / SD on the analysis scale."""
    if ERRORBAR == 'ci':
        return row['ci_lo'], row['ci_hi']
    half = row['sem'] * (np.sqrt(row['n_slices']) if ERRORBAR == 'sd' else 1.0)
    if LOG:
        return row['mean_fc'] * np.exp(-half), row['mean_fc'] * np.exp(half)
    return row['mean_fc'] - half, row['mean_fc'] + half


def _between_p(group, feature, condition):
    r = BETWEEN[condition]
    r = r[(r['feature'] == feature) & (r['comparison'] == f'{group} vs {REFERENCE}')]
    return np.nan if r.empty else p_reported(r.iloc[0])


def _hashes(p):
    """Between-group marks: # / ## / ### (the reference paper's convention for post hoc tests)."""
    return stars(p).replace('*', '#')


def _pad(y_lo, y_hi):
    """Axis limits around the data: multiplicative padding on log, from 0 on linear."""
    if LOG:
        return y_lo / 1.35, y_hi * 1.35
    return 0.0, y_hi * 1.08


def _data_range(feature, condition='drug'):
    """Padded axis limits covering every group's error bar and slices (and 1), so axes match across groups."""
    lo = hi = 1.0
    for g in GROUPS:
        row = _within_row(g, feature, condition)
        v = _slice_fc(g, feature, condition)
        if row is None or len(v) == 0:
            continue
        e_lo, e_hi = _err(row)
        lo, hi = min(lo, e_lo, v.min()), max(hi, e_hi, v.max())
    return _pad(lo, hi)


def _level(y_hi, k, span):
    """Height of the k-th significance bracket above y_hi."""
    return y_hi * STEP ** (0.25 + k) if LOG else y_hi + span * (0.06 + 0.12 * k)


def _extend(y_hi, k, span):
    """Axis top after drawing k brackets."""
    return y_hi * STEP ** (k + 0.6) if LOG else y_hi + span * (0.12 * k + 0.08)


def _bracket(ax, x0, x1, y, label, span):
    """Significance bracket at height y."""
    tip = y / (STEP ** 0.18) if LOG else y - span * 0.03
    text_y = y * (STEP ** 0.06) if LOG else y + span * 0.01
    ax.plot([x0, x0, x1, x1], [tip, y, y, tip],
            lw=1.3, color='#222222', zorder=6, clip_on=False)
    ax.text((x0 + x1) / 2, text_y, label, ha='center', va='bottom',
            fontsize=9, color='#222222', fontweight='bold', zorder=7)


def _finish_axis(ax, y_lo, y_hi):
    ax.axhline(1.0, color='grey', ls=':', lw=1.0, alpha=0.7, zorder=1)
    ax.grid(True, axis='y', ls='--', alpha=0.28, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', labelsize=8)
    if LOG:
        ax.set_yscale('log')
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel('Fold change, drug / baseline (log scale)', fontsize=8)
        # plain tick labels (0.5, 1, 2, 5)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=12))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: f'{v:g}' if v >= 1 else f'{v:.2g}'))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    else:
        ax.set_ylim(y_lo, y_hi)
        ax.set_ylabel('Fold change, drug / baseline', fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(6))


def _grid(n, w, h):
    n_cols = 3
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * w, n_rows * h),
                             constrained_layout=True)
    axes = np.array(axes).flatten()
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    return fig, axes


# --- Main figure: all groups, within- and between-group tests ---

def plot_between(condition='drug'):
    rng   = np.random.default_rng(7)
    feats = PRIMARY_FEATURES

    fig, axes = _grid(len(feats), 3.6, 3.8)
    fig.suptitle('Fold change during psilocybin relative to each slice\'s own baseline',
                 fontsize=11, fontweight='bold')
    fig.supxlabel(
        f'Bar = {MEAN_NAME}, error bar = {ERR_LABEL}, dots = slices; n = slices/animals.\n'
        f'* {TEST} (BH-FDR across the {N_PRIMARY} parameters); '
        f'# one-way ANOVA with Holm–Šidák vs {REFERENCE}.',
        fontsize=8, color='#444444')

    for i, feat in enumerate(feats):
        ax = axes[i]
        y_lo, y_hi = _data_range(feat, condition)
        span = y_hi - y_lo
        for xi, g in enumerate(GROUPS):
            row = _within_row(g, feat, condition)
            v   = _slice_fc(g, feat, condition)
            if row is None or len(v) == 0:
                continue
            m = row['mean_fc']
            lo, hi = _err(row)
            ax.bar(xi, m, width=0.6, color=COLORS.get(g, '#888'),
                   edgecolor=COLORS.get(g, '#888'), alpha=0.85, zorder=2)
            ax.errorbar(xi, m, yerr=[[m - lo], [hi - m]], fmt='none',
                        ecolor='#222222', elinewidth=1.3, capsize=4, zorder=5)
            ax.scatter(xi + rng.uniform(-0.16, 0.16, len(v)), v, s=28,
                       color='#1A1A1A', alpha=0.9, edgecolors='white',
                       linewidths=0.6, zorder=4)
            # within-group mark (vs no change) just above the bar's highest point
            p = p_reported(row)
            if not pd.isna(p) and p < ALPHA:
                top = max(hi, v.max())
                ax.text(xi, top * STEP ** 0.05 if LOG else top + span * 0.015, stars(p),
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        color='#222222', zorder=7)

        k = 0
        for xi, g in enumerate(GROUPS):
            if g == REFERENCE:
                continue
            p = _between_p(g, feat, condition)
            if pd.isna(p) or p >= ALPHA:
                continue
            _bracket(ax, GROUPS.index(REFERENCE), xi, _level(y_hi, k, span),
                     _hashes(p), span)
            k += 1
        if k:
            y_hi = _extend(y_hi, k, span)

        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels([f'{g}\nn = {N_LABEL[g]}' for g in GROUPS], fontsize=8)
        _finish_axis(ax, y_lo, y_hi)
        ax.set_title(fig_label(feat), fontsize=9, fontweight='bold', color='#222222')

    fname = f'between_group_{condition}_slice_bars{SUFFIX}'
    fig.savefig(os.path.join(PLOTS_DIR, f'{fname}.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  saved  plots_slice/{fname}.png')


# --- Supplementary: one group, Baseline vs Drug ---

def plot_within(group):
    color = COLORS.get(group, '#4878CF')
    rng   = np.random.default_rng(42)
    feats = PRIMARY_FEATURES
    conds = ['baseline', 'drug']

    fig, axes = _grid(len(feats), 3.7, 3.6)
    fig.suptitle(f'{group} — fold change during psilocybin relative to each slice\'s own baseline '
                 f'(n = {N_LABEL[group]})', fontsize=11, fontweight='bold')
    fig.supxlabel(
        f'Bar = {MEAN_NAME}, error bar = {ERR_LABEL}, dots = slices; baseline is 1 by definition.  '
        f'* {TEST} (BH-FDR across the {N_PRIMARY} parameters).  Y-axes match across groups.',
        fontsize=8, color='#444444')

    face = {'baseline': 'white', 'drug': color}
    edge = {'baseline': '#333333', 'drug': color}

    for i, feat in enumerate(feats):
        ax = axes[i]
        y_lo, y_hi = _data_range(feat, 'drug')
        span = y_hi - y_lo
        for xi, cond in enumerate(conds):
            if cond == 'baseline':
                ax.bar(xi, 1.0, width=0.55, color=face[cond],
                       edgecolor=edge[cond], linewidth=1.5, zorder=2)
                continue
            row = _within_row(group, feat, cond)
            v   = _slice_fc(group, feat, cond)
            if row is None or len(v) == 0:
                continue
            m = row['mean_fc']
            lo, hi = _err(row)
            ax.bar(xi, m, width=0.55, color=face[cond], edgecolor=edge[cond],
                   linewidth=1.5, zorder=2)
            ax.errorbar(xi, m, yerr=[[m - lo], [hi - m]], fmt='none',
                        ecolor='#222222', elinewidth=1.4, capsize=4,
                        capthick=1.4, zorder=5)
            ax.scatter(xi + rng.uniform(-0.15, 0.15, len(v)), v, s=30,
                       color='#111111', alpha=0.9, edgecolors='white',
                       linewidths=0.6, zorder=4)

        row = _within_row(group, feat, 'drug')
        p = np.nan if row is None else p_reported(row)
        any_sig = not pd.isna(p) and p < ALPHA
        if any_sig:
            _bracket(ax, 0, 1, _level(y_hi, 0, span), stars(p), span)
        # always leave room for one bracket, so every group's axis is identical
        y_hi = _extend(y_hi, 1, span)

        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c.capitalize() for c in conds], fontsize=8)
        _finish_axis(ax, y_lo, y_hi)
        ax.set_title(fig_label(feat), fontsize=9,
                     fontweight='bold' if any_sig else 'normal',
                     color=color if any_sig else '#555555')

    fig.legend(handles=[
        mpatches.Patch(facecolor='white', edgecolor='#333333', label='Baseline'),
        mpatches.Patch(facecolor=color, edgecolor=color, label='Drug'),
    ], loc='outside upper right', fontsize=8, framealpha=0.9, ncol=2)

    fig.savefig(os.path.join(PLOTS_DIR, f'{group}_slice_bars{SUFFIX}.png'),
                bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'  saved  plots_slice/{group}_slice_bars{SUFFIX}.png')


if __name__ == '__main__':
    print(f'\nFigures (primary features only, {SCALE} scale):')
    plot_between('drug')
    for g in GROUPS:
        plot_within(g)
    print(f'\nAll figures → {PLOTS_DIR}/')
    print('Exploratory features are deliberately not plotted — they carry '
          'uncorrected p-values.')
