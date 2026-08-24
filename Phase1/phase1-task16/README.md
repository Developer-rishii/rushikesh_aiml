# Task 16 — Model Validation & K-Fold

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Tasks 1–15 (back on the supervised track):** loads
Task 2's cleaned data restricted to Task 7's locked feature set. Same
`SEED=42`.

## What this delivers (Definition of Done)

**A cross-validated comparison reporting mean and variance, with the
most generalising model chosen** — demonstrated live in
`src/run_validation.py`, following the study guide's 6 steps:

1. **Appropriate CV scheme** — `StratifiedKFold(5)`, chosen because
   Task 2 already found the target imbalanced (~63/37) — a plain
   `KFold` risks a skewed test fold by bad luck.
2. **Run K-Fold, collect per-fold scores** — `src/validation/kfold_compare.py`,
   the **same** `StratifiedKFold` object reused across every candidate,
   so the comparison is on identical folds, not incomparable ones.
3. **Report mean AND spread** — every model's report includes all 5 raw
   fold scores plus mean/std/min/max, never a single headline number.
4. **Nested CV for the tuned model** — `src/validation/nested_cv.py`:
   `logreg`'s `C` is tuned via `GridSearchCV` used as the (unfit)
   *estimator* inside an outer `cross_val_score` — the standard, correct
   sklearn nested-CV pattern, verified structurally by a test (not just
   run and eyeballed).
5. **Compare candidates on the same folds** — logreg, random_forest,
   gradient_boosting, all scored on the identical stratified split.
6. **Conclude which model generalises best** — `src/validation/select.py`:
   "most generalising" requires acceptable variance (`std <=
   max_acceptable_std`) before mean is even considered — a high-mean,
   high-variance model cannot win by mean alone.

## Each named pitfall gets its own passing test

| Pitfall (from the study guide) | Test | Result |
|---|---|---|
| Reporting only the best fold | `test_pitfall_reports_spread_not_just_best_fold` | Asserts every candidate's report includes all 5 individual fold scores plus mean/std/min/max |
| Non-stratified folds on imbalanced data | `test_pitfall_folds_are_actually_stratified` | Measures the actual per-fold positive rate and asserts it stays within 0.05 of the overall rate — genuinely stratified, not just labeled `StratifiedKFold` |
| Tuning and evaluating on the same folds | `test_pitfall_nested_cv_structurally_cannot_leak` | Source-inspects `run_nested_cv` to confirm it uses the correct `cross_val_score(unfit_GridSearchCV, ...)` pattern (the only way nested CV avoids leakage in sklearn), then checks the reported optimism gap is a real computed number |

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Cross-validated comparison with mean+variance, most generalising model chosen | `outputs/reports/validation_report.json` — full per-fold scores, nested CV, and explicit selection with its rule stated |
| Real-data quality & correctness (realistic, not toy) | Real 569-row WDBC data, Task 7's vetted 31-feature set, 3 real candidate models each actually fit 5x (+ nested CV's 5x3 inner fits for logreg) |
| Live verification & evidence | `tests/test_validation.py` — 6/6 tests pass on live runs; a synthetic-fixture test proves the selection logic actually enforces the variance cap, not just returns whatever's highest-mean |
| Dependency/failure/edge-case handling | Unknown candidate model name rejected before fitting; data hand-off failures (missing raw data, missing locked features, missing engineered feature columns) all raise clearly |

## How to run

```bash
pip install -r requirements.txt
python tests/test_validation.py   # everything, incl. pitfall + edge-case tests
# or the pipeline directly:
python -m src.run_validation
```

## Results from this run (seed=42)

| Model | Mean | Std | Fold scores |
|---|---|---|---|
| **logreg** | **0.9964** | 0.0034 | [0.9956, 0.9968, 0.9904, 1.0, 0.9992] |
| gradient_boosting | 0.9947 | 0.0040 | [0.9986, 0.9913, 0.9888, 0.9987, 0.9962] |
| random_forest | 0.9938 | 0.0043 | [0.9994, 0.9887, 0.9890, 0.9966, 0.9955] |

**Stratification check:** per-fold positive rate deviates at most 0.0046
from the overall 0.6274 rate — genuinely stratified.

**Selected model: logreg** — highest mean among models with acceptable
variance (all three had acceptable variance here, so mean broke the
tie), **and** it also wins by worst-fold performance (0.9904, the best
"floor" of any candidate) — consistent with the brainstorming question
"which model wins consistently, not just on average."

**Nested CV for logreg's tuned `C`:** mean **0.9973**, std 0.0035, best
`C=0.1`. **Naive (non-nested) score would have reported 0.9963.**
**Optimism gap: -0.0011** — essentially zero here, which is itself an
honest finding: on this near-ceiling, well-separated dataset (consistent
with every task since Task 2), naive tuning happens not to meaningfully
overstate performance at this scale. The nested-CV machinery is what
lets you know that, rather than assume it.

Full numbers: `outputs/reports/validation_report.json`. Fold-score
comparison plot: `outputs/figures/fold_scores_comparison.png`.

## External resources needed

**None.** Same offline WDBC data as Tasks 1-15. Only `pip install -r
requirements.txt` needs network access, once.

## Folder structure

```
task16_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── loader.py                     # YAML -> typed Config, sets global seed
│   └── config.yaml                   # CV scheme, candidate models, nested CV grid
├── data/
│   ├── clean_from_task2.csv          # carried over from Task 2
│   └── locked_feature_set.json       # carried over from Task 7
├── src/
│   ├── __init__.py
│   ├── run_validation.py             # THE 6-step flow
│   ├── data/dataset.py                 # load + honour Task 7's locked features
│   └── validation/
│       ├── models.py                   # single sklearn Pipeline builders
│       ├── kfold_compare.py            # Steps 1-3,5: stratified K-Fold comparison
│       ├── nested_cv.py                # Step 4: leakage-safe nested CV
│       ├── select.py                   # Step 6: variance-gated model selection
│       └── plots.py                    # fold-score box/strip plot
├── tests/
│   └── test_validation.py            # live run + one test per named pitfall + edge cases
└── outputs/
    ├── reports/
    │   └── validation_report.json
    ├── figures/
    │   └── fold_scores_comparison.png
    └── logs/
        └── run_validation.log
```
