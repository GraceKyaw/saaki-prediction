# saaki-prediction

Prediction of sepsis-associated acute kidney injury (SA-AKI).

## Status

Fresh working repository. Related earlier work — the reproduction of published
SA-AKI models, organised by AKI definition (`sa1/`, `sa3/`) — lives in
[GraceKyaw/reproducing-sa-aki](https://github.com/GraceKyaw/reproducing-sa-aki).

## Layout

```
data/         # local only — never committed (see .gitignore)
notebooks/    # exploration, modelling, results
sql/          # cohort and feature extraction queries
src/          # reusable code
figures/      # generated plots and diagrams
documents/    # write-up, reports, reading notes, references
```

## Data

Patient-level data is MIMIC-derived and requires credentialed PhysioNet access.
Nothing under `data/` is tracked: `.gitignore` excludes `*.csv`, `*.parquet`,
model binaries and prediction files. Only code, SQL, aggregate metrics and
figures belong in version control.

To reproduce, extract the cohort locally with the queries in `sql/` and place
the outputs in `data/`.
