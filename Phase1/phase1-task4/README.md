# Task 4 — The Pre-Processing Protocol

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Tasks 1–3:** loads Task 2/3's cleaned WDBC dataset
(`data/clean_from_task2.csv`), same `SEED=42`. The raw WDBC feature set is
all-numeric with zero missing values (a curated benchmark), so to make
categorical encoding and imputation genuinely demonstrable — not
hypothetical — `src/data/dataset.py::enrich_dataframe` adds one realistic
categorical field (`sample_processing_site`) and injects ~4% missingness
into 3 numeric columns before the split. This is declared plainly in the
config and the report, not hidden.

## What this delivers (Definition of Done)

A **fitted, leak-free preprocessing pipeline, reusable at train and
inference time**, demonstrated live — not just described:

- `src/preprocessing/pipeline.py` is the core deliverable: builds a
  `ColumnTransformer` (impute → scale numerics, impute → one-hot encode
  categoricals), fits it **only on `X_train`**, and provides
  `save_preprocessor` / `load_preprocessor` for inference reuse.
- `src/run_protocol.py` runs the whole thing end-to-end **and then
  reloads the saved artifact from disk and transforms new rows with it**
  — proving inference-time reuse actually works, not asserting it does.

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Reproducible env, correct split, smoke-test w/ logged metrics | `SEED=42` via `configs/loader.py`; stratified 70/15/15 split with index-overlap guard in `src/data/dataset.py`; `outputs/reports/preprocessing_report.json` + `outputs/logs/run_protocol.log` from a real run |
| Real-data quality & correctness (realistic, not toy) | Real 569-row WDBC data, plus realistic categorical + missing-value enrichment so encoding/imputation isn't vacuous |
| Live verification & evidence | `tests/test_preprocessing.py` — 7/7 tests pass on a live run, including a leakage guard proven against a *synthetic real leak* (see below), not just a happy path |
| Dependency/failure/edge-case handling | Empty-dataframe, missing-artifact-file, and all-NaN-column cases all raise specific errors; every stage wrapped with `sys.exit(1)` on failure |

## Each named pitfall gets its own passing test

| Pitfall (from the brief) | Test | Result |
|---|---|---|
| Fitting the scaler on all data (leakage) | `test_pitfall_scaler_not_fit_on_all_data` | Scaler's learned mean matches train-only, not train+val+test |
| Train/serve preprocessing drift | `test_pitfall_no_train_serve_drift` | Reloaded artifact's output is bit-identical to the original fitted object's |
| Dropping rows with missing values | `test_pitfall_missing_values_imputed_not_dropped` | Row count unchanged after transform; zero NaNs remain |
| (Bonus) Does the leakage guard actually work? | `test_leakage_guard_raises_on_synthetic_leak` | Deliberately fits a preprocessor on train+val, confirms `verify_no_leakage` catches and rejects it — the guard was tested against a real failure, not just a pass |

## How to run

```bash
pip install -r requirements.txt
python tests/test_preprocessing.py   # everything, incl. pitfall tests + edge cases
# or just the protocol:
python -m src.run_protocol
```

## Results from this run (seed=42)

- Train/val/test: 398 / 85 / 86 rows.
- 30 numeric + 1 categorical feature → 33 output columns after encoding.
- 45 missing values in the training data going in, **0 rows dropped**, **0
  NaNs remaining** after transform.
- Leakage check: 0 index overlap between train/val; fitted scaler's mean
  matches train-only, confirmed not the train+val combined mean.
- Reloaded preprocessor's output on 5 fresh inference rows is identical to
  the original in-memory object's output.

Full numbers: `outputs/reports/preprocessing_report.json`.

## External resources needed

**None.** Same offline WDBC data carried over from Tasks 1–3. Only
`pip install -r requirements.txt` needs network access, once.

## Folder structure

```
task4_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── loader.py                    # YAML -> typed Config, sets global seed
│   └── config.yaml                  # paths, split, enrichment, preprocessing choices
├── data/
│   └── clean_from_task2.csv         # carried over from Task 2/3
├── src/
│   ├── __init__.py
│   ├── run_protocol.py              # orchestrates fit -> verify -> save -> reload -> reuse
│   ├── data/
│   │   └── dataset.py                # load + realistic enrichment + stratified split
│   └── preprocessing/
│       └── pipeline.py               # THE core deliverable: fit/transform/verify/save/load
├── tests/
│   └── test_preprocessing.py         # live run + one test per named pitfall + edge cases
└── outputs/
    ├── preprocessors/
    │   └── fitted_preprocessor.joblib
    ├── reports/
    │   └── preprocessing_report.json
    └── logs/
        └── run_protocol.log
```
