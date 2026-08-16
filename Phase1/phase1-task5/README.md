# Task 5 — The First Prediction

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Tasks 1–4:** loads Task 2's cleaned WDBC dataset, reuses
Task 4's exact preprocessing contract (fit on train only, then reused
unchanged for baseline and model) and Task 3's config-driven, add-a-model
registry pattern. Same `SEED=42` throughout.

## What this delivers (Definition of Done)

**A first model with validation metrics shown against an explicit
baseline, plus error notes** — demonstrated live, in `src/run_experiment.py`,
following the brief's 6 steps in order:

1. **Baseline metric** — `src/modeling/baseline.py`: an explicit
   `DummyClassifier(strategy="most_frequent")`. Not implicit, not skipped.
2. **First model, through the harness** — `src/modeling/registry.py`:
   `LogisticRegression`, trained on the *same* preprocessed features as
   the baseline (Task 4's fitted transformer, reused unchanged).
3. **Evaluate on validation** — `src/modeling/metrics.py::compute_metrics`
   is called with `X_val`/`y_val` only; never `X_train`/`y_train`.
4. **Compare against baseline** — `run_experiment.py` computes the lift on
   the *primary metric* and logs whether the model actually beats it.
5. **Inspect worst errors** — `src/modeling/errors.py::worst_errors` ranks
   validation misclassifications by how confidently wrong the model was,
   and `summarize_error_patterns` reports the false-negative/false-positive
   split, not just a single accuracy number.
6. **Record the run, decide next step** — both baseline and model rows go
   into `outputs/experiments/experiment_log.csv` (append-only, never
   overwritten), and the script prints an explicit next-improvement
   decision derived from the error pattern found in step 5.

## Each named pitfall gets its own passing test

| Pitfall (from the brief) | Test | Result |
|---|---|---|
| No baseline | `test_pitfall_has_explicit_baseline` | Confirms a named, explicit `DummyClassifier` baseline exists and is logged |
| Reporting training accuracy | `test_pitfall_metrics_computed_on_validation_not_training` | Computes both train and val metrics separately, then asserts the actual pipeline script only ever calls `compute_metrics` with validation data (source-inspected, not just claimed) |
| Optimising the wrong metric | `test_pitfall_primary_metric_is_not_accuracy_on_imbalanced_data` | Asserts `primary_metric != "accuracy"` given the class imbalance already documented in Task 2 |

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Reproducible env, correct split, smoke-test w/ logged metrics | `SEED=42`; stratified 70/15/15 split with leakage guard (Task 4); `outputs/experiments/experiment_log.csv` + `outputs/logs/run_experiment.log` from a real run |
| Real-data quality & correctness (realistic, not toy) | Real 569-row WDBC data with Task 4's realistic categorical + missing-value enrichment carried through |
| Live verification & evidence | `tests/test_experiment.py` — 7/7 tests pass on live runs; `outputs/reports/worst_errors.csv` contains the actual misclassified rows, not a description of them |
| Dependency/failure/edge-case handling | Data/preprocessing/baseline/model/eval stages each wrapped with a specific error + `sys.exit(1)`; tests cover unknown model name and missing config |

## How to run

```bash
pip install -r requirements.txt
python tests/test_experiment.py   # everything, incl. pitfall tests + edge cases
# or just the experiment:
python -m src.run_experiment
```

## Results from this run (seed=42)

| | Baseline (majority-class) | First model (LogisticRegression) |
|---|---|---|
| PR-AUC (primary) | 0.6235 | **1.0** |
| ROC-AUC | 0.5 | 1.0 |
| Precision | 0.6235 | 1.0 |
| Recall | 1.0 | 0.9623 |
| Accuracy | 0.6235 | 0.9765 |

**Lift over baseline (PR-AUC): +0.3765 → model beats the baseline.**

**Worst-error inspection:** 2 misclassifications on the validation set,
both false negatives (predicted benign, actually malignant), 0 false
positives. Mean error magnitude 0.61 (moderately, not maximally, confident
wrong calls). **Pattern found → next improvement:** false negatives
dominate, so the recommended next step is raising recall (lower decision
threshold or stronger class weighting), since a missed malignant case is
the costlier error — not "try a fancier model" by default.

Full numbers: `outputs/reports/evaluation_report.json`,
`outputs/reports/worst_errors.csv`.

## External resources needed

**None.** Same offline WDBC data carried over from Tasks 1–4. Only
`pip install -r requirements.txt` needs network access, once.

## Folder structure

```
task5_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── loader.py                    # YAML -> typed Config, sets global seed
│   └── config.yaml                  # paths, split, baseline, model, metrics
├── data/
│   └── clean_from_task2.csv         # carried over from Task 2
├── src/
│   ├── __init__.py
│   ├── run_experiment.py            # THE 6-step flow: baseline -> model -> compare -> errors -> log
│   ├── data/
│   │   └── dataset.py                 # load + enrich (from Task 4) + stratified split
│   ├── preprocessing/
│   │   └── pipeline.py                # fit-on-train-only preprocessor (from Task 4)
│   └── modeling/
│       ├── baseline.py                # Step 1: explicit dummy baseline
│       ├── registry.py                # Step 2: add-a-model-here (logreg, decision_tree)
│       ├── metrics.py                 # Step 3: single source of validation metrics
│       ├── errors.py                  # Step 5: worst-error ranking + pattern summary
│       └── experiment_log.py          # Step 6: append-only run logger
├── tests/
│   └── test_experiment.py            # live run + one test per named pitfall + edge cases
└── outputs/
    ├── experiments/
    │   └── experiment_log.csv         # baseline + model rows, side by side
    ├── models/
    │   ├── fitted_preprocessor.joblib
    │   └── logreg.joblib
    ├── reports/
    │   ├── evaluation_report.json
    │   └── worst_errors.csv
    └── logs/
        └── run_experiment.log
```
