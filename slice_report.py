"""Combined 17-feature table and effect-retention table (drug vs baseline). Writes summary_drug.csv and effect_pattern_drug.csv."""

import os
import numpy as np
import pandas as pd

from slice_data import (RESULTS_DIR, GROUPS, REFERENCE, EVENT_RATE, ALPHA,
                        SCALE, feat_label, CWD)
from slice_robust import (PRIMARY_FEATURES, within_group, between_group, stars,
                          fmt_effect, MEAN_NAME, EFFECT, EFFECT_NAME, p_reported)

# Display order, by measurement domain
FEATURE_ORDER = [
    ('Integrated signal', ["Curve - dat AUC", "Curve - df AUC", "Curve - dff AUC"]),
    ('Kinetics', ["Curve - Duration of visualized event overlay",
                  "Curve - Duration 50% to 50% based on averge dF/F",
                  "Curve - Duration 10% to 10% based on averge dF/F",
                  "Curve - Rising duration 10% to 90% based on averge dF/F",
                  "Curve - Decaying duration 90% to 10% based on averge dF/F"]),
    ('Amplitude', ["Curve - Max Df", "Curve - Max Dff"]),
    ('Shape', ["Basic - Perimeter (only for 2D video)", "Basic - Circularity"]),
    ('Network', ["Network - number of events in the same location",
                 "Network - number of events in the same location with similar size only",
                 "Network - maximum number of events appearing at the same time"]),
    ('Activity', [EVENT_RATE]),
]


def build_summary(cond='drug'):
    w, b = within_group(cond), between_group(cond)
    rows = []
    for block, feats in FEATURE_ORDER:
        for feat in feats:
            fr = w[w['feature'] == feat]
            if fr.empty:
                continue
            family = fr.iloc[0]['family']
            row = {'block': block, 'feature': feat_label(feat), 'family': family,
                   'condition': cond, 'scale': SCALE}
            for g in GROUPS:
                r = fr[fr['group'] == g]
                if r.empty:
                    row[g] = 'n/a'
                    continue
                r = r.iloc[0]
                p = p_reported(r)
                row[f'{g}_n'] = r['n_slices']
                row[f'{g}_fc'] = r['mean_fc']
                row[f'{g}_ci_lo'] = r['ci_lo']
                row[f'{g}_ci_hi'] = r['ci_hi']
                row[f'{g}_p'] = p
                row[g] = (f'{r["mean_fc"]:.2f} '
                          f'({r["ci_lo"]:.2f}-{r["ci_hi"]:.2f}) {stars(p)}')
            for g in GROUPS:
                if g == REFERENCE:
                    continue
                rb = b[(b['feature'] == feat) &
                       (b['comparison'] == f'{g} vs {REFERENCE}')]
                key = f'{g} vs {REFERENCE}'
                if rb.empty:
                    row[key] = 'n/a'
                    continue
                rb = rb.iloc[0]
                p = p_reported(rb)
                row[f'{key}_{EFFECT}'] = rb['effect_vs_ref']
                row[f'{key}_p'] = p
                row[key] = f'{fmt_effect(rb["effect_vs_ref"])} {stars(p)}'
            rows.append(row)
    return pd.DataFrame(rows)


def print_summary(df, cond):
    ns = {g: int(df[f'{g}_n'].dropna().max()) if f'{g}_n' in df else 0
          for g in GROUPS}
    print(f'\n{"=" * 108}')
    print(f'{cond.upper()}  —  {MEAN_NAME} vs own baseline (95% CI), '
          f'n = slices  [{"  ".join(f"{g} n={ns[g]}" for g in GROUPS)}]')
    print('=' * 108)
    tags = {'primary':       'PRIMARY — BH-FDR within group, Holm-Sidak between groups',
            'exploratory':   'EXPLORATORY — UNCORRECTED p, hypothesis-generating only',
            'f0_confounded': 'RAW-FLUORESCENCE UNITS — confounded by F0 drift, not interpreted'}
    for family in ['primary', 'exploratory', 'f0_confounded']:
        tag = tags[family]
        print(f'\n## {tag}')
        print(f'{"feature":24s}{"WT":>22s}{"AV":>22s}{"IP":>22s}'
              f'{"AV vs WT":>10s}{"IP vs WT":>10s}')
        block = None
        for _, r in df[df['family'] == family].iterrows():
            if r['block'] != block:
                block = r['block']
                print(f'-- {block}')
            print(f'{r["feature"]:24s}{r["WT"]:>22s}{r["AV"]:>22s}{r["IP"]:>22s}'
                  f'{r[f"AV vs {REFERENCE}"]:>10s}{r[f"IP vs {REFERENCE}"]:>10s}')
    print(f'\n* p<0.05  ** p<0.01  *** p<0.001    between-group values are the {EFFECT_NAME} '
          f'vs {REFERENCE}')


def effect_pattern(cond='drug', ref=REFERENCE):
    """keeps% = group mean change / reference mean change x 100, where change is log FC (log scale)
    or FC - 1 (linear scale). Unstable when the reference effect is small."""
    w, b = within_group(cond), between_group(cond)
    sig = w[(w['group'] == ref) & (w['family'] != 'f0_confounded') &
            (w.apply(p_reported, axis=1) < ALPHA)]
    others = [g for g in GROUPS if g != ref]
    rows = []
    for _, s in sig.sort_values('p_raw').iterrows():
        feat = s['feature']
        row = {'feature': feat_label(feat), 'family': s['family'],
               f'{ref}_fc': s['mean_fc']}
        for g in others:
            r = w[(w['group'] == g) & (w['feature'] == feat)]
            if r.empty:
                continue
            r = r.iloc[0]
            row[f'{g}_fc'] = r['mean_fc']
            row[f'{g}_keeps_%'] = (r['mean_change'] / s['mean_change'] * 100
                                   if s['mean_change'] != 0 else np.nan)
            rb = b[(b['feature'] == feat) & (b['comparison'] == f'{g} vs {ref}')]
            if not rb.empty:
                row[f'{g}_effect'] = rb.iloc[0]['effect_vs_ref']
                row[f'{g}_vs_{ref}_p'] = p_reported(rb.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def print_effect_pattern(df, cond, ref=REFERENCE):
    others = [g for g in GROUPS if g != ref]
    print(f'\n{"=" * 108}')
    print(f'{cond.upper()}  —  do {" and ".join(others)} follow {ref}\'s response? '
          f'(features where {ref} responded)')
    print('=' * 108)
    hdr = f'{"feature":24s}{"fam":>6s}{ref:>7s}'
    for g in others:
        hdr += f'{g:>7s}{"keeps":>8s}{EFFECT[:5]:>7s}{"sig":>5s}'
    print(hdr)
    for _, r in df.iterrows():
        line = f'{r["feature"]:24s}{r["family"][:4]:>6s}{r[f"{ref}_fc"]:7.2f}'
        for g in others:
            line += (f'{r.get(f"{g}_fc", np.nan):7.2f}'
                     f'{r.get(f"{g}_keeps_%", np.nan):7.0f}%'
                     f'{fmt_effect(r.get(f"{g}_effect", np.nan)):>7s}'
                     f'{stars(r.get(f"{g}_vs_{ref}_p", np.nan)):>5s}')
        print(line)


if __name__ == '__main__':
    out = os.path.join(CWD, RESULTS_DIR)
    os.makedirs(out, exist_ok=True)
    for cond in ['drug']:
        df = build_summary(cond)
        print_summary(df, cond)
        df.to_csv(os.path.join(out, f'summary_{cond}.csv'), index=False)

        pat = effect_pattern(cond)
        print_effect_pattern(pat, cond)
        pat.to_csv(os.path.join(out, f'effect_pattern_{cond}.csv'), index=False)

    print(f'\nSaved → {out}/summary_drug.csv, effect_pattern_drug.csv')
