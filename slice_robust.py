"""
Drug-vs-baseline statistics on per-slice fold changes, on the scale set by SCALE in slice_data.py.
linear: mean fold change; within group, one-sample t-test vs 1; between groups, difference in mean FC.
log:    geometric mean fold change; one-sample t-test of log FC vs 0; ratio of geometric means.
Between group: one-way ANOVA, then Holm-Sidak vs WT using the ANOVA's pooled variance (CIs use the same variance).
Primary features: BH-FDR across the three within each group (within-group tests); between-group tests carry the
Holm-Sidak-adjusted p only (one correction, as in the reference paper).
(Wilcoxon / Kruskal-Wallis + Dunn are used only if NORMALITY_A > 0 and Shapiro-Wilk fails.)
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from slice_data import (
    slice_table, RESULTS_DIR, GROUPS, REFERENCE, FEATURES, EVENT_RATE,
    ALPHA, NORMALITY_A, AGGREGATOR, SCALE, feat_label,
)

# --- Feature families ---

PRIMARY_FEATURES = [
    "Curve - dff AUC",                                # integrated signal
    "Curve - Duration of visualized event overlay",   # kinetics
    EVENT_RATE,                                       # activity
]
# Raw-fluorescence units: scale with resting F0, which is lower in drug recordings.
F0_CONFOUNDED = ["Curve - Max Df", "Curve - dat AUC", "Curve - df AUC"]
EXPLORATORY = [f for f in FEATURES
               if f not in PRIMARY_FEATURES and f not in F0_CONFOUNDED]


def _family(feat):
    if feat in PRIMARY_FEATURES:
        return 'primary'
    if feat in F0_CONFOUNDED:
        return 'f0_confounded'
    return 'exploratory'


FAMILY_FEATURES = {'primary': PRIMARY_FEATURES, 'exploratory': EXPLORATORY,
                   'f0_confounded': F0_CONFOUNDED}
FAMILY_TAGS = {
    'primary':       (f'PRIMARY (within group: BH-FDR across these {len(PRIMARY_FEATURES)}; '
                      'between group: Holm-Sidak)'),
    'exploratory':   'EXPLORATORY (uncorrected p — hypothesis-generating only)',
    'f0_confounded': 'RAW-FLUORESCENCE UNITS — confounded by F0 drift, not interpreted',
}

CI_LEVEL = 0.95

# --- Analysis scale ---

if SCALE not in ('linear', 'log'):
    raise ValueError(f"SCALE must be 'linear' or 'log', not {SCALE!r}")
NULL      = 0.0 if SCALE == 'log' else 1.0             # "no change" on the analysis scale
EFFECT    = 'ratio' if SCALE == 'log' else 'difference'  # between-group effect vs reference
MEAN_NAME = 'geometric mean fold change' if SCALE == 'log' else 'mean fold change'
EFFECT_NAME = ('ratio of geometric mean fold changes' if SCALE == 'log'
               else 'difference in mean fold change')


def fmt_effect(x):
    """Between-group effect as text: ratio 0.82 or difference -0.23."""
    return f'{x:.2f}' if EFFECT == 'ratio' else f'{x:+.2f}'


# --- Statistical helpers ---

def _holm_sidak(pvals):
    """Holm-Sidak step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        a = 1.0 - (1.0 - p[idx]) ** (m - rank)
        running = max(running, a)          # enforce monotonicity
        adj[idx] = min(running, 1.0)
    return adj


def _dunn_vs_control(groups_vals, control_idx=0):
    """Dunn's test vs the control group (tie-corrected, Holm-adjusted); returns raw and adjusted p."""
    all_vals = np.concatenate(groups_vals)
    ranks    = stats.rankdata(all_vals)
    N        = len(all_vals)

    sizes, mean_ranks, pos = [], [], 0
    for v in groups_vals:
        k = len(v)
        sizes.append(k)
        mean_ranks.append(ranks[pos:pos + k].mean())
        pos += k

    _, counts = np.unique(all_vals, return_counts=True)
    ties = (counts ** 3 - counts).sum()
    sigma2 = (N * (N + 1) / 12.0) - ties / (12.0 * (N - 1))

    raw = []
    for i in range(len(groups_vals)):
        if i == control_idx:
            continue
        se = np.sqrt(sigma2 * (1.0 / sizes[i] + 1.0 / sizes[control_idx]))
        z  = (mean_ranks[i] - mean_ranks[control_idx]) / se if se > 0 else 0.0
        raw.append(2 * stats.norm.sf(abs(z)))
    adj = multipletests(raw, method='holm')[1] if raw else []
    return raw, list(adj)


def stars(p):
    if pd.isna(p) or p >= ALPHA:
        return 'ns'
    return '***' if p < 0.001 else '**' if p < 0.01 else '*'


def _fc_vals(group, feature, cond):
    """Per-slice fold changes on the analysis scale (log: non-positive values dropped)."""
    m = ((slice_table['group'] == group) & (slice_table['feature'] == feature) &
         (slice_table['condition'] == cond))
    v = slice_table.loc[m, 'fc'].dropna().values
    if SCALE == 'log':
        return np.log(v[v > 0]), len(v) - int((v > 0).sum())
    return v, 0


def _back(x):
    """Analysis scale -> fold change."""
    return np.exp(x) if SCALE == 'log' else x


def _summary(vals):
    """Mean fold change (geometric if log), its CI, and the SEM on the analysis scale."""
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan
    m = vals.mean()
    if n < 2:
        return _back(m), np.nan, np.nan, np.nan
    sem = vals.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.5 + CI_LEVEL / 2, n - 1) * sem
    return _back(m), _back(m - h), _back(m + h), sem


# --- Within group: one-sample test of FC vs no change ---

def within_group(cond='drug'):
    rows = []
    for gname in GROUPS:
        for feat in FEATURES:
            vals, dropped = _fc_vals(gname, feat, cond)
            if len(vals) < 3:
                continue

            shapiro_p = stats.shapiro(vals).pvalue if np.ptp(vals) > 0 else np.nan
            normal = (not np.isnan(shapiro_p)) and shapiro_p >= NORMALITY_A
            t_stat, p_t = stats.ttest_1samp(vals, NULL)
            p_w = np.nan
            if len(vals) >= 5 and np.ptp(vals) > 0:
                try:
                    p_w = stats.wilcoxon(vals - NULL).pvalue
                except ValueError:
                    pass
            use_t = normal or np.isnan(p_w)
            mean_fc, lo, hi, sem = _summary(vals)
            change = vals.mean() - NULL

            rows.append({
                'group': gname, 'feature': feat, 'label': feat_label(feat),
                'family': _family(feat), 'scale': SCALE,
                'condition': cond, 'n_slices': len(vals), 'n_dropped': dropped,
                'mean_fc': mean_fc, 'ci_lo': lo, 'ci_hi': hi,
                'mean_change': change, 'sem': sem,
                'shapiro_p': shapiro_p,
                'test_used': 'one-sample t-test' if use_t else 'Wilcoxon signed-rank',
                't_stat': t_stat, 'p_ttest': p_t, 'p_wilcoxon': p_w,
                'p_raw': p_t if use_t else p_w, 'p_fdr': np.nan,
                'cohens_dz': change / vals.std(ddof=1) if vals.std(ddof=1) > 0 else np.nan,
            })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    # BH-FDR within primary features only
    for gname in GROUPS:
        m = (res['group'] == gname) & (res['family'] == 'primary')
        if m.sum() > 0:
            res.loc[m, 'p_fdr'] = multipletests(res.loc[m, 'p_raw'].values,
                                                method='fdr_bh')[1]
    return res


# --- Between group: one-way ANOVA on FC + post hoc vs reference ---

def between_group(cond='drug', ref=REFERENCE):
    others = [g for g in GROUPS if g != ref]
    rows = []
    for feat in FEATURES:
        vals, present = [], []
        for g in [ref] + others:
            v, _ = _fc_vals(g, feat, cond)
            if len(v) >= 3:
                vals.append(v)
                present.append(g)
        if len(vals) < 2 or present[0] != ref:
            continue

        normal = all((stats.shapiro(v).pvalue if np.ptp(v) > 0 else 0.0) >= NORMALITY_A
                     for v in vals)
        # pooled variance and df from the one-way ANOVA (used by post hoc and CIs)
        df_anova = sum(len(v) for v in vals) - len(vals)
        mse = sum(((v - v.mean()) ** 2).sum() for v in vals) / df_anova
        if normal:
            omni_stat, omni_p = stats.f_oneway(*vals)
            omni, post = 'one-way ANOVA', 'Holm-Sidak'
            raw = [2 * stats.t.sf(abs(v.mean() - vals[0].mean())
                                  / np.sqrt(mse * (1 / len(v) + 1 / len(vals[0]))), df_anova)
                   for v in vals[1:]]
            adj = _holm_sidak(raw)
        else:
            omni_stat, omni_p = stats.kruskal(*vals)
            omni, post = 'Kruskal-Wallis', 'Dunn'
            raw, adj = _dunn_vs_control(vals, control_idx=0)

        mean_ref = _back(vals[0].mean())
        for j, g in enumerate(present[1:]):
            v = vals[j + 1]
            mean_g, lo, hi, _ = _summary(v)
            # effect vs reference and its CI (same pooled variance as the post hoc):
            # log: ratio of geometric means; linear: difference in mean fold change
            diff = v.mean() - vals[0].mean()
            se = np.sqrt(mse * (1 / len(v) + 1 / len(vals[0])))
            h = stats.t.ppf(0.5 + CI_LEVEL / 2, df_anova) * se
            rows.append({
                'condition': cond, 'feature': feat, 'label': feat_label(feat),
                'family': _family(feat), 'scale': SCALE,
                'comparison': f'{g} vs {ref}',
                f'n_{ref}': len(vals[0]), 'n_group': len(v),
                f'mean_fc_{ref}': mean_ref, 'mean_fc_group': mean_g,
                'ci_lo_group': lo, 'ci_hi_group': hi,
                'effect_type': EFFECT, 'effect_vs_ref': _back(diff),
                'effect_ci_lo': _back(diff - h), 'effect_ci_hi': _back(diff + h),
                'omnibus_test': omni, 'omnibus_stat': omni_stat, 'omnibus_p': omni_p,
                'posthoc_test': post, 'p_raw': raw[j], 'p_posthoc': adj[j],
            })
    return pd.DataFrame(rows)


# --- Reporting ---

def p_reported(r):
    """Reported p: primary features carry the BH-FDR p (within group) or the Holm-Sidak p
    (between group); exploratory features carry the uncorrected p."""
    if r['family'] != 'primary':
        return r['p_raw']
    return r['p_posthoc'] if 'comparison' in r.index else r['p_fdr']


def _sig(r):
    p = p_reported(r)
    return stars(p), p


def print_within(res, cond):
    print(f'\n{"=" * 104}')
    print(f'WITHIN-GROUP — {cond.upper()} vs own baseline. {MEAN_NAME.capitalize()} '
          f'({int(CI_LEVEL * 100)}% CI), slice as unit, {SCALE} scale')
    print('=' * 104)
    for family in ['primary', 'exploratory', 'f0_confounded']:
        tag = FAMILY_TAGS[family]
        print(f'\n-- {tag} ' + '-' * max(0, 100 - len(tag)))
        print(f'{"feature":24s}{"grp":>4s}{"n":>4s}{"FC":>9s}{"95% CI":>18s}'
              f'{"test":>22s}{"p":>9s}  sig')
        for feat in FAMILY_FEATURES[family]:
            for g in GROUPS:
                r = res[(res['group'] == g) & (res['feature'] == feat) &
                        (res['condition'] == cond)]
                if r.empty:
                    continue
                r = r.iloc[0]
                s, p = _sig(r)
                ci = f'({r["ci_lo"]:.2f}–{r["ci_hi"]:.2f})'
                print(f'{r["label"]:24s}{g:>4s}{r["n_slices"]:4d}{r["mean_fc"]:9.2f}'
                      f'{ci:>18s}{r["test_used"]:>22s}{p:9.4f}  {s}')


def print_between(res, cond, ref=REFERENCE):
    print(f'\n{"=" * 104}')
    print(f'BETWEEN-GROUP — {cond.upper()}, vs {ref}. {EFFECT_NAME.capitalize()} '
          f'({int(CI_LEVEL * 100)}% CI), {SCALE} scale')
    print('=' * 104)
    for family in ['primary', 'exploratory', 'f0_confounded']:
        tag = FAMILY_TAGS[family]
        print(f'\n-- {tag} ' + '-' * max(0, 100 - len(tag)))
        print(f'{"feature":24s}{"comparison":>10s}{EFFECT[:5]:>8s}{"95% CI":>18s}'
              f'{"omnibus":>16s}{"p_omni":>8s}{"p":>9s}  sig')
        for feat in FAMILY_FEATURES[family]:
            sub = res[(res['feature'] == feat) & (res['condition'] == cond)]
            for _, r in sub.iterrows():
                s, p = _sig(r)
                ci = f'({fmt_effect(r["effect_ci_lo"])}–{fmt_effect(r["effect_ci_hi"])})'
                print(f'{r["label"]:24s}{r["comparison"]:>10s}{fmt_effect(r["effect_vs_ref"]):>8s}'
                      f'{ci:>18s}{r["omnibus_test"]:>16s}{r["omnibus_p"]:8.3f}'
                      f'{p:9.4f}  {s}')


if __name__ == '__main__':
    out = os.path.join(os.getcwd(), RESULTS_DIR)
    os.makedirs(out, exist_ok=True)
    print(f'Per-slice summary: {AGGREGATOR}   |   analysis scale: {SCALE} fold change')
    print(f'Primary features ({len(PRIMARY_FEATURES)}): '
          + ', '.join(feat_label(f) for f in PRIMARY_FEATURES))
    print(f'Exploratory ({len(EXPLORATORY)}): '
          + ', '.join(feat_label(f) for f in EXPLORATORY))

    for cond in ['drug']:
        w = within_group(cond)
        b = between_group(cond)
        print_within(w, cond)
        print_between(b, cond)
        w.to_csv(os.path.join(out, f'robust_within_{cond}.csv'), index=False)
        b.to_csv(os.path.join(out, f'robust_between_{cond}.csv'), index=False)

    print(f'\nSaved → {out}/robust_within_*.csv, robust_between_*.csv')
    print('Primary features carry BH-FDR p-values within group and Holm-Sidak p-values between '
          'groups;\nexploratory features carry uncorrected p-values and must be labelled '
          'exploratory wherever they are reported.')
