# saaki-prediction

Prediction of sepsis-associated acute kidney injury (SA-AKI) from the first
24 hours after sepsis onset in the ICU, comparing static and temporal
representations of the same patient data.

## Status

MSc dissertation research project (2026). The code and results here are the
version submitted for assessment; later edits are limited to documentation.

This is retrospective research on a public critical-care database, not a
validated or deployed clinical tool.

An earlier, separate piece of work reproduced published SA-AKI models,
organised by AKI definition (`sa1/`, `sa3/`):
[GraceKyaw/reproducing-sa-aki](https://github.com/GraceKyaw/reproducing-sa-aki).
This repository builds new models on a fresh cohort rather than extending
that reproduction.

## Overview

**Question.** Does describing a patient's first 24 hours after sepsis onset
temporally (as an hourly sequence, or as first/last/trend values) predict
later AKI better or earlier than static aggregates of the same hours?

**Cohort.** Built from MIMIC-IV v3.1 with the queries in `sql/`, using
Sepsis-3 for sepsis onset and KDIGO stage ≥ 1 for AKI:

1. **14,826 ICU stays** meet the SQL criteria: Sepsis-3, age 18–90, an ICU
   stay of at least 48 hours, no AKI before sepsis onset, and at least 24
   hours between sepsis onset and ICU discharge.
2. **12,972 stays** remain after keeping only each patient's first ICU stay,
   by sepsis onset time. One stay was then dropped for not having a full 24
   hours of data when re-checked from its timestamps.
3. **5,069 stays** remain after excluding the 7,902 whose AKI occurred within
   the 24-hour observation window. Of these, 60.5% develop AKI afterwards.

With one stay per patient, every stay-level row is a different patient, so
the stay-level train/test split and evaluation are effectively patient-level
for this cohort.

**Representations** (built in `notebooks/preprocessing.ipynb`):

| | representation | rows | model |
|---|---|---|---|
| B1 | min, max, mean and last of each feature | one per stay | XGBoost, static FFNN |
| B2 | hourly sequence, 24 × 22 features | one per stay | GRU |
| B3 | first, last and trend per non-overlapping 24-hour window | several per stay | temporal-feature FFNN (exploratory) |
| B4 | first, last and trend over the B1 window | one per stay | static FFNN with trend features |

B4 differs from B1 only in feature representation, which isolates its effect.
B3 reassesses a patient window by window with a window-level label (24.3%
positive), so it is reported separately rather than ranked against the others.

**Validation.** Each model is run under two conditions: no imbalance handling,
and class weighting. Each uses a stratified 80/20 train/test split
(grouped by patient for B3), 50-trial Optuna hyperparameter searches over
five inner cross-validation folds, and a single evaluation on the held-out
test set. Winsorization, imputation and scaling are fitted on training data
only. Reported with bootstrap 95% confidence intervals (resampling whole
patients for B3): AUPRC (primary), AUROC, recall, specificity, precision, F1,
calibration, decision curves, subgroup performance and lead time.

## Principal findings

Held-out test set, no imbalance handling, 1,014 stays (prevalence 60.5%):

| model | AUPRC [95% CI] | AUROC [95% CI] |
|---|---|---|
| XGBoost (B1) | 0.736 [0.695, 0.773] | 0.649 [0.614, 0.687] |
| GRU (B2) | 0.710 [0.670, 0.751] | 0.630 [0.596, 0.665] |
| static FFNN (B1) | 0.700 [0.659, 0.739] | 0.614 [0.575, 0.650] |
| static FFNN, trend features (B4) | 0.693 [0.653, 0.736] | 0.609 [0.574, 0.644] |

- XGBoost on static aggregates scored highest, but every confidence interval
  overlaps, and all models are only moderately above the no-skill AUPRC of
  0.605.
- The temporal comparison gave mixed findings. Trend features (B4) did not
  change AUPRC compared with aggregates (B1). Recall was higher without
  imbalance handling, but specificity was lower. The GRU did not flag AKI
  measurably earlier than the static FFNN: median lead time was 20 h against
  19 h, with overlapping intervals.
- Class weighting traded recall for specificity without improving AUPRC.

## Limitations and trade-offs

- **Calibration.** Class weighting made the predicted probabilities less
  well calibrated in every model. For XGBoost, expected calibration error rose
  from 0.035 with no imbalance handling to 0.119 with class weighting.
- **Age subgroups.** For XGBoost with no imbalance handling, specificity
  (correctly clearing patients who did not develop AKI) fell with age: 0.51 in
  the 18–40 band, 0.25 in 40–65 and 0.12 in 65–90. Older patients without AKI
  were therefore flagged much more often. This figure is for that one
  configuration; the other models' subgroup results are in
  `notebooks/model_comparison.ipynb`.
- **Comparability.** B3 is labelled per window and has a different positive
  rate (24.3% against 60.5%), so its AUPRC is not directly comparable with the
  stay-level models and it is reported separately.
- **Scope.** Single-centre retrospective data, a single held-out test set and
  no external validation. Results describe this research cohort, not
  clinical performance.

Full tables, calibration and figures are produced by
`notebooks/model_comparison.ipynb`; plots are in `figures/`.

## Layout

```
sql/          # cohort definition, feature extraction and checks (MIMIC-IV, BigQuery)
notebooks/    # preprocessing, five model training notebooks, cross-model comparison
src/          # shared fit/transform and evaluation functions
figures/      # generated plots
documents/    # placeholder for public documentation
```

`src/cluster_bootstrap_addition.py` holds the patient-level bootstrap
functions on their own; the same functions are in `src/evaluation_utils.py`,
which is what the notebooks import.

## Data

Patient-level data is MIMIC-IV-derived and requires credentialed PhysioNet
access. None of the source data or derived extracts are included here.

Without credentialed access, visitors can inspect the code, SQL, notebook
methodology and figures, but cannot reproduce the data-dependent experiments
from this repository alone.

Nothing under `data/` is tracked: `.gitignore` excludes the whole folder, as
well as `*.csv`, `*.parquet`, model binaries and prediction files elsewhere.
Only code, SQL, aggregate metrics and figures belong in version control.

## Reproducing

1. With MIMIC-IV access, run `sql/feature_extraction.sql` and save the result
   as `data/raw_extraction.csv`. The other SQL files are checks on the cohort,
   features and windows.
2. Run the notebooks from inside `notebooks/`, in this order:
   1. `preprocessing.ipynb`: writes the four representations to `data/`
   2. `model_training_xgboost.ipynb`
   3. `model_training_static_ffnn.ipynb`
   4. `model_training_gru.ipynb`
   5. `model_training_temporal_ffnn.ipynb`
   6. `model_training_static_trend.ipynb`
   7. `model_comparison.ipynb`: reads every `*_results.json` and
      `*_feature_importance.json` written to `data/` by the training
      notebooks, and writes the comparison tables to `data/` and plots to
      `figures/`

**Environment.** Python 3.13 with pandas, NumPy, pyarrow, scikit-learn,
XGBoost 3.3, Keras 3.15, Optuna 4.9, matplotlib and seaborn. imbalanced-learn
is needed only for the unused SMOTE utility in
`src/preprocessing_fit_transform.py`. Random seeds are fixed at 42, but neural
network training can still vary slightly between machines.
