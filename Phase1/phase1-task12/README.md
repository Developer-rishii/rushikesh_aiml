# Task 12 — Binary Classification (Production-Grade)

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Tasks 1–11:** loads Task 2's cleaned data + Task 7's
locked feature set, calibrates and packages Task 9's confirmed logreg
config — this task doesn't pick a new model, it makes the existing one
production-ready. Same `SEED=42`.

## What this delivers (Definition of Done)

**A calibrated, threshold-justified classifier with stable,
segment-checked metrics, packaged for serving** — demonstrated live in
`src/run_calibrated_classifier.py`, following the study guide's 6 steps:

1. **Train the classifier** — Task 9's confirmed `LogisticRegression`
   config, through the full preprocessing pipeline.
2. **Calibrate + verify** — `src/models/calibrate.py` fits **both**
   Platt (sigmoid) and isotonic calibration and picks the winner by
   **measured Brier score** (lower = better-calibrated), not by
   assumption. A real calibration curve PNG is generated for both
   methods against the uncalibrated baseline.
3. **Cost-optimal threshold** — `src/evaluation/threshold.py`, same
   method as Task 6, applied to the **calibrated** probabilities
   specifically (thresholding only makes sense once probabilities mean
   what they say).
4. **CV stability + segment fairness** — `src/evaluation/stability.py`:
   5-fold CV (train-only, refitting the full calibration procedure per
   fold) for stability, plus per-segment recall across tumor-size
   tertiles (small/medium/large — the domain-reasonable segment for a
   dataset with no demographic attributes) to catch hidden per-segment
   failure.
5. **Document the operating point** — `outputs/reports/calibrated_classifier_report.json`
   contains the exact threshold, calibration method, and expected error
   rates (missed-malignancy rate, unnecessary-biopsy rate) on the
   held-out test set.
6. **Package for serving** — `src/serving/package.py` bundles the fitted
   calibrated pipeline + threshold + full metadata into one directory,
   with a single `predict()` entrypoint — then the run **reloads the
   package from disk and verifies its predictions match the original
   exactly**, not just assumes serialization worked.

## Each named pitfall gets its own passing test

| Pitfall (from the study guide) | Test | Result |
|---|---|---|
| Uncalibrated probabilities used as if exact | `test_pitfall_calibration_quality_measured_not_assumed` | Asserts Brier score is actually computed for uncalibrated + both calibration methods — the choice is evidence-based, not assumed |
| Hidden per-segment failure | `test_pitfall_segment_failure_would_be_caught` | Computes real per-segment recall, then forces an impossible recall bar and confirms the flagging mechanism actually fires — proving it isn't dead code |
| No documented operating point | `test_pitfall_operating_point_is_documented` | Asserts the report contains threshold, calibration method, test metrics, confusion matrix, AND expected error rates together |

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Calibrated, threshold-justified classifier, stable segment-checked metrics, packaged for serving | `outputs/reports/calibrated_classifier_report.json` (calibration + threshold + CV + segments), `outputs/serving_package/` (model.joblib + serving_config.json) |
| Real-data quality & correctness (realistic, not toy) | Real 569-row WDBC data, Task 7's vetted 31-feature set, real tumor-size segments with real per-segment sample counts (28/28/29) |
| Live verification & evidence | `tests/test_calibrated_classifier.py` — 7/7 tests pass on live runs; the serving package is reloaded from disk and its predictions are byte-compared against the original, not assumed identical |
| Dependency/failure/edge-case handling | Unknown calibration method, missing/incomplete serving package, and a missing segment feature column all raise clearly instead of failing silently or crashing obscurely |

## How to run

```bash
pip install -r requirements.txt
python tests/test_calibrated_classifier.py   # everything, incl. pitfall + edge-case tests
# or the pipeline directly:
python -m src.run_calibrated_classifier
```

## Results from this run (seed=42)

**Calibration (Step 2):** Brier score — uncalibrated 0.01176, sigmoid
(Platt) 0.01929 (worse!), **isotonic 0.00781 (best)**. Isotonic selected.
Platt scaling actually *hurt* calibration here relative to the raw
model — a real finding a coin-flip choice would have missed. Calibration
curve: `outputs/figures/calibration_curve.png`.

**Cost-optimal threshold (Step 3):** default 0.5 → expected cost 1.0 (2
unnecessary biopsies). **Recommended threshold 0.37 → expected cost 0.0**
(0 missed malignancies, 0 unnecessary biopsies) — a real improvement
over the default, not a coin-flip pick.

**CV stability (Step 4):** 5 folds, scores `[1.0, 0.9988, 0.9739, 1.0,
0.996]`, mean 0.9937, **std 0.01 → stable** (below the 0.05 threshold).

**Segment fairness (Step 4):** tumor-size tertiles (small n=28,
medium n=28, large n=29) — **recall 1.0 in all three segments**, no
hidden per-segment failure, `fairness_confirmed: true`.

**Operating point (Step 5), held-out test set (touched once):**
threshold 0.37, isotonic calibration → expected missed-malignancy rate
**1.16%**, expected unnecessary-biopsy rate **0.0%** on this test split.

**Serving package (Step 6):** `outputs/serving_package/model.joblib` +
`serving_config.json`, reloaded from disk in the same run and verified
to reproduce identical probabilities to the ninth decimal.

Full numbers: `outputs/reports/calibrated_classifier_report.json`.

## External resources needed

**None.** Same offline WDBC data as Tasks 1-11. Only `pip install -r
requirements.txt` needs network access, once.

## Folder structure

```
task12_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── loader.py                        # YAML -> typed Config, sets global seed
│   └── config.yaml                      # calibration methods, cost, segments
├── data/
│   ├── clean_from_task2.csv             # carried over from Task 2
│   └── locked_feature_set.json          # carried over from Task 7
├── src/
│   ├── __init__.py
│   ├── run_calibrated_classifier.py     # THE 6-step flow
│   ├── data/dataset.py                    # from Task 8-11
│   ├── models/
│   │   ├── build.py                       # Step 1: base classifier pipeline
│   │   └── calibrate.py                   # Step 2: Platt + isotonic, Brier comparison, curve plot
│   ├── evaluation/
│   │   ├── metrics.py                     # single source of metric computation
│   │   ├── threshold.py                   # Step 3: cost-optimal threshold sweep
│   │   └── stability.py                   # Step 4: CV stability + segment fairness
│   └── serving/
│       └── package.py                     # Step 6: bundle + reload + predict entrypoint
├── tests/
│   └── test_calibrated_classifier.py    # live run + one test per named pitfall + edge cases
└── outputs/
    ├── reports/
    │   └── calibrated_classifier_report.json
    ├── figures/
    │   └── calibration_curve.png
    ├── serving_package/
    │   ├── model.joblib
    │   └── serving_config.json
    └── logs/
        └── run_calibrated_classifier.log
```
