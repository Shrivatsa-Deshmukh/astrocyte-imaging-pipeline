# Astrocyte Imaging Pipeline

> **Work in progress.** The analysis and results below are preliminary and may change.

Analysis of AQuA2-segmented astrocyte calcium imaging (3,008 events, 48 recordings)
to test how **psilocybin** alters astrocyte signaling in wild-type, 5-HT2A-antagonist
(volinanserin) and IP3R2 conditional-knockout slices.

## Key findings

- Psilocybin increased astrocyte **integrated calcium signal by 25%** and **event
  duration by 20%** (both P < 0.001).
- Both effects were **abolished in IP3R2 knockouts** (P < 0.01 vs WT).
- **5-HT2A blockade** trended in the same direction but did not differ
  significantly from WT.

| Parameter | WT (n = 10) | AV (n = 8) | IP (n = 6) | AV − WT | IP − WT |
|---|---|---|---|---|---|
| Integrated signal (AUC ΔF/F) | 1.25*** | 1.20* | 1.02 | −0.05 | −0.23## |
| Event duration | 1.20*** | 1.15* | 0.94 | −0.05 | −0.26## |
| Event rate | 1.65* | 1.20 | 1.70 | −0.45 | 0.05 |

Fold change during psilocybin vs each slice's own baseline (1.0 means no change); n = slices.
\* vs baseline (one-sample t-test, BH-FDR); # vs WT (one-way ANOVA + Holm–Šidák).
One symbol: P < 0.05; two: P < 0.01; three: P < 0.001.

## Approach

- **Slice as the unit of analysis:** events are averaged per slice and condition,
  not pooled as independent observations.
- **Within-slice normalization:** drug value ÷ the same slice's baseline.
- **Statistics:** three primary parameters (integrated signal, duration, event
  rate) with FDR correction; 95% CIs for all effects.

## Pipeline

| File | Role |
|---|---|
| `slice_data.py` | Settings; loads AQuA2 CSVs and builds per-slice fold changes |
| `slice_robust.py` | Within- and between-group statistics |
| `slice_report.py` | Summary tables |
| `slice_barplots.py` | Figures |


The scripts expect AQuA2 exports under `Output__/<group>/<animal>/sliceN_<condition>_AQuA2.csv`.
This data is not included in the repository. Groups and slices are set in
`DATA_CONFIG` in `slice_data.py`.
