"""Load AQuA2 CSVs and build the per-slice table: one value per slice, condition and feature, as a fold change vs that slice's baseline."""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Config ---

MIN_FRAME      = None                # Starting Frame window; None = full recording
MAX_FRAME      = None
OUTPUT_DIR     = 'Output__'          # AQuA2 CSVs (read-only)
RESULTS_DIR    = 'Output_slice'
CHANNEL_SUFFIX = 'Ch1'
CWD            = os.getcwd()

ALPHA          = 0.05
ERRORBAR       = 'ci'                # figure error bars: 'ci' (95% CI), 'sem' or 'sd'
NORMALITY_A    = 0.0                 # 0.0 = parametric tests throughout; 0.05 = rank tests when Shapiro-Wilk fails
REFERENCE      = 'WT'

AGGREGATOR     = 'mean'              # per-slice summary: 'mean' or 'median'
SCALE          = 'linear'            # fold-change analysis scale: 'linear' (default) or 'log'

DATA_CONFIG = {
    'WT': {
        'path': 'WT',
        'drug_suffix': 'psi',
        'slices': {'data1': [2, 3], 'data2': [1, 2, 3, 4], 'data3': [1, 2, 3, 4]},   # data1/slice1 removed: see README
    },
    'AV': {
        'path': 'Antagonist- Volinanserin',
        'drug_suffix': 'psi+antag',
        'slices': {'data1': [1, 2, 3, 4], 'data2': [1, 2, 3, 4]},
    },
    'IP': {
        'path': 'IP3R2 cKO',
        'drug_suffix': 'psi',
        'slices': {'data1': [2, 4, 5], 'data2': [1, 2, 3]},
    },
}

ALL_FEATURES = [
    "Curve - Max Df", "Curve - Max Dff",
    "Curve - dat AUC", "Curve - df AUC", "Curve - dff AUC",
    "Basic - Perimeter (only for 2D video)", "Basic - Circularity",
    "Curve - Duration of visualized event overlay",
    "Curve - Duration 50% to 50% based on averge dF/F",
    "Curve - Duration 10% to 10% based on averge dF/F",
    "Curve - Rising duration 10% to 90% based on averge dF/F",
    "Curve - Decaying duration 90% to 10% based on averge dF/F",
    "Network - number of events in the same location",
    "Network - number of events in the same location with similar size only",
    "Network - maximum number of events appearing at the same time",
]

EVENT_RATE = "Event rate (events per slice)"   # pseudo-feature: events per recording
FEATURES   = ALL_FEATURES + [EVENT_RATE]       # event area is not analysed

FEATURE_LABELS = {
    "Curve - Max Df":                                                    "Max ΔF",
    "Curve - Max Dff":                                                   "Max ΔF/F",
    "Curve - dat AUC":                                                   "AUC (raw)",
    "Curve - df AUC":                                                    "AUC (ΔF)",
    "Curve - dff AUC":                                                   "AUC (ΔF/F)",
    "Basic - Perimeter (only for 2D video)":                             "Perimeter",
    "Basic - Circularity":                                               "Circularity",
    "Curve - Duration of visualized event overlay":                      "Duration (overlay)",
    "Curve - Duration 50% to 50% based on averge dF/F":                  "Duration (50–50%)",
    "Curve - Duration 10% to 10% based on averge dF/F":                  "Duration (10–10%)",
    "Curve - Rising duration 10% to 90% based on averge dF/F":           "Rise time (10–90%)",
    "Curve - Decaying duration 90% to 10% based on averge dF/F":         "Decay time (90–10%)",
    "Network - number of events in the same location":                   "Co-location",
    "Network - number of events in the same location with similar size only": "Co-loc. (same size)",
    "Network - maximum number of events appearing at the same time":     "Overlapping events",
    EVENT_RATE:                                                          "Event rate",
}

CONDITIONS = ['baseline', 'drug', 'washout']
GROUPS     = list(DATA_CONFIG)

_DROP_COLS = {'Channel', 'Index', 'Curve - P Value on max Dff (-log10)', 'Curve - Decay tau'}


def feat_label(feat):
    return FEATURE_LABELS.get(feat, feat.split(' - ')[-1][:35])


# Figure panel titles: the names used in Methods_Results.md (tables and printouts keep the short labels)
FIG_LABELS = {
    "Curve - dff AUC":                              "Integrated signal (AUC ΔF/F)",
    "Curve - Duration of visualized event overlay": "Event duration",
}


def fig_label(feat):
    return FIG_LABELS.get(feat, feat_label(feat))


# --- Loading ---

def _file_paths(config):
    ch = f'_{CHANNEL_SUFFIX}' if CHANNEL_SUFFIX else ''
    for subfolder, nums in config['slices'].items():
        folder = os.path.join(CWD, OUTPUT_DIR, config['path'], subfolder)
        for n in nums:
            yield (
                os.path.join(folder, f'slice{n}_baseline_AQuA2{ch}.csv'),
                os.path.join(folder, f'slice{n}_{config["drug_suffix"]}_AQuA2{ch}.csv'),
                os.path.join(folder, f'slice{n}_washout_AQuA2{ch}.csv'),
                subfolder, n,
            )


def _load_file(path):
    """Read one AQuA2 CSV as an event table (one row per event)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, header=None).set_index(0).T.reset_index(drop=True)
    df = df.drop(columns=list(_DROP_COLS & set(df.columns)), errors='ignore')
    df = df.apply(pd.to_numeric, errors='coerce')
    # Drop AQuA2's blank placeholder column (written when no events are detected).
    if 'Starting Frame' in df.columns:
        df = df[df['Starting Frame'].notna()].copy()
    if 'Starting Frame' in df.columns and (MIN_FRAME is not None or MAX_FRAME is not None):
        lo = -np.inf if MIN_FRAME is None else MIN_FRAME
        hi = np.inf if MAX_FRAME is None else MAX_FRAME
        df = df[df['Starting Frame'].between(lo, hi)].copy()
    return df if not df.empty else pd.DataFrame()


def build_slice_table():
    """One row per (group, slice, condition, feature); fc = value / same slice's baseline value."""
    rows = []
    for gname, cfg in DATA_CONFIG.items():
        for bp, dp, wp, subfolder, snum in _file_paths(cfg):
            b, d, w = _load_file(bp), _load_file(dp), _load_file(wp)
            if b.empty or d.empty:
                print(f'  skipping {gname}/{subfolder}/slice{snum}: missing baseline or drug')
                continue

            frames = {'baseline': b, 'drug': d}
            if not w.empty:
                frames['washout'] = w          # loaded, not analysed
            base_mean = {}
            for feat in ALL_FEATURES:
                base_mean[feat] = (getattr(b[feat], AGGREGATOR)(skipna=True)
                                   if feat in b.columns else np.nan)
            base_mean[EVENT_RATE] = float(len(b))

            for cond, df in frames.items():
                for feat in FEATURES:
                    if feat == EVENT_RATE:
                        m, sd, n_ev = float(len(df)), np.nan, len(df)
                    elif feat in df.columns:
                        vals = df[feat].dropna()
                        if vals.empty:
                            continue
                        m = getattr(vals, AGGREGATOR)()
                        sd, n_ev = vals.std(ddof=1), len(vals)
                    else:
                        continue

                    bm = base_mean.get(feat, np.nan)
                    fc = np.nan if (pd.isna(bm) or bm == 0) else m / bm
                    rows.append({
                        'group':       gname,
                        'data_folder': subfolder,
                        'slice_num':   snum,
                        'slice_id':    f'{subfolder}_slice{snum}',
                        'condition':   cond,
                        'feature':     feat,
                        'n_events':    n_ev,
                        'raw_mean':    m,
                        'raw_sd':      sd,
                        'fc':          fc,
                    })

    tab = pd.DataFrame(rows)
    return tab.replace([np.inf, -np.inf], np.nan)


# --- Build on import ---

_win = 'all frames' if MIN_FRAME is None and MAX_FRAME is None else f'frames {MIN_FRAME}-{MAX_FRAME}'
print(f'Loading slice-level table ({AGGREGATOR} per slice, {_win}, n = slices)...')
slice_table = build_slice_table()

if slice_table.empty:
    raise SystemExit('No data loaded — check OUTPUT_DIR and DATA_CONFIG paths.')

# events per recording = the largest per-feature event count (features with no missing values)
_counts = (slice_table[slice_table['condition'] == 'drug']
           .groupby(['group', 'slice_id'])['n_events'].max()
           .groupby('group').agg(['count', 'sum']))
for _g in GROUPS:
    if _g in _counts.index:
        _r = _counts.loc[_g]
        print(f'  {_g}: {int(_r["count"])} slices, {int(_r["sum"])} drug events')


def save_slice_table():
    """Write slice_means_long.csv."""
    out = os.path.join(CWD, RESULTS_DIR)
    os.makedirs(out, exist_ok=True)
    slice_table.to_csv(os.path.join(out, 'slice_means_long.csv'), index=False)
    print(f'Saved slice table → {out}/')


if __name__ == '__main__':
    save_slice_table()
    print('\nThis file only prepares data. Run slice_robust.py for the analysis.')
