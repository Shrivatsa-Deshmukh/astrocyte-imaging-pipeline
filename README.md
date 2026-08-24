# Astrocyte Imaging Pipeline

Post-processing pipeline for astrocyte calcium imaging data exported from [AQuA2](https://github.com/yu-lab-vt/AQuA2). Handles per-slice normalization, quantile regression, and visualization across experimental groups.

---

## Notebooks

**`analysis.ipynb`** — run first
- Loads and transposes AQuA2 CSV exports
- Normalizes features to baseline via per-slice fold-change
- Between-group quantile regression at Q50 / Q75 / Q90 (WT as reference)
- Within-group quantile regression at Q50 / Q75 / Q90 (Baseline as reference, per group)
- Exports normalized CSVs, raw CSVs, frame-binned CSVs, and event count CSVs
- Plot functions available for between-group dot plots, within-group quantile trajectory plots, and timepoint dot plots

**`analysis_time.ipynb`** — run after main
- Loads normalized CSVs from the main notebook
- Bins events into 10-frame windows and summarizes within each slice before aggregating — equal slice weight regardless of event count
- Time-series plots: group median ± MAD, all groups overlaid, Drug and Washout side by side

---

## Setup

**Dependencies:** `numpy pandas matplotlib scipy statsmodels`

Organize AQuA2 exports under `Output__/`:

```
Output__/
├── WT/
│   ├── data1/
│   │   ├── slice1_baseline_AQuA2_Ch1.csv
│   │   ├── slice1_psi_AQuA2_Ch1.csv
│   │   ├── slice1_washout_AQuA2_Ch1.csv
│   │   └── ...
│   └── data2/
├── Antagonist- Volinanserin/
├── IP3R2 cKO/
└── CalEx/
```

Key parameters (configuration cell):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MIN_FRAME` | `20` | Start frame |
| `MAX_FRAME` | `100` | End frame |
| `CHANNEL_SUFFIX` | `'Ch1'` | AQuA2 channel suffix; set to `''` if unused |
| `CONTROL_GROUP` | `'WT'` | Reference group for between-group regression |
| `BIN_SIZES` | `[5, 10, 20]` | Frame bin widths for exported CSVs |

---

## Experimental Groups

| Group | Description |
|-------|-------------|
| WT | Wild Type + Psilocybin |
| AV | 5-HT2A antagonist (Volinanserin) + Psilocybin |
| IP | IP3R2 conditional knockout + Psilocybin |
| CE | CalEx + Psilocybin |

Three conditions per group: Baseline → Drug → Washout. A slice is included only if all three condition files are present. 

---

## Normalization

Per-slice fold-change relative to that slice's own baseline median:

```
normalized_value = event_feature_value / slice_baseline_median
```

Both drug and washout values use the same denominator — y = 1.0 throughout means no change from pre-drug baseline.

---

## Statistical Methods

Two quantile regression analyses at Q50 / Q75 / Q90.

**Between-group** — pools all events per condition, Group as predictor, WT as reference:
```
feature_FC ~ C(Group)
```
Coefficient = difference in q-th percentile relative to WT. Identifies pathway-dependent effects.

**Within-group** — pools events per condition within each group, Condition as predictor, Baseline as reference:
```
feature_FC ~ C(Condition)
```
Coefficient = shift in q-th percentile relative to own baseline. 

---

## Key Results

Two findings from the drug condition. Between-group values are Δ FC vs WT; within-group values are Δ FC vs own baseline.

### Finding 1 — Pathway-dependent: extreme event population requiring 5-HT2A/IP3R2

WT generates events that are larger, more synchronised, and more spatially recurrent than baseline. Both AV and IP fail to generate this population — not active suppression, but failure to achieve the WT increase.

| Feature (vs WT, Drug) | AV Q50 | AV Q75 | AV Q90 | IP Q50 | IP Q75 | IP Q90 |
|---------|--------|--------|--------|--------|--------|--------|
| Max simultaneous events | −0.91 *** | −2.13 *** | −6.41 *** | −0.95 *** | −2.23 *** | −6.70 *** |
| Area | −0.24 *** | −0.70 *** | −1.02 ** | ns | −1.16 *** | −2.32 *** |
| Co-location (similar size) | ns | −0.50 *** | −2.00 *** | ns | ns | −2.00 *** |

Max simultaneous events is the strongest signal — WT increases +1.13 at Q50 to +6.93 at Q90 within-group, with near-identical suppression in both AV and IP at Q90.

### Finding 2 — Pathway-independent: integrated calcium activity

dF/F AUC and Duration increase above baseline in both WT and AV with comparable magnitude, producing no between-group difference. Only detectable via within-group analysis.

| Feature (Drug vs Baseline) | WT Q50 | WT Q75 | WT Q90 | AV Q50 | AV Q75 | AV Q90 |
|---------|--------|--------|--------|--------|--------|--------|
| dF/F AUC | +0.15 *** | +0.27 *** | +0.48 *** | +0.18 *** | +0.18 ** | +0.35 ** |
| Duration (overlay) | +0.11 *** | +0.14 ** | +0.27 *** | ~ | ~ | ~ |

IP is excluded from this table: with only 90 drug events across 6 slices, the within-group model could not be computed for either feature due to insufficient non-missing values per condition.
