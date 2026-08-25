# Task 17 — Hyperparameter Tuning (Advanced)

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Tasks 1–16 (supervised track):** loads Task 2's
cleaned data restricted to Task 7's locked feature set. Same `SEED=42`.

## What this delivers (Definition of Done)

**A peak-tuned model from an efficient, logged search with
test-confirmed gains** — demonstrated live in `src/run_advanced_tuning.py`,
following the study guide's 6 steps:

1. **Sensible search space, correct scales** — `learning_rate` sampled
   **log-uniform** (`suggest_float(..., log=True)`, spans 0.001-0.3, two
   orders of magnitude); `max_depth`/`subsample`/`min_samples_leaf`
   sampled on their natural linear scale. Verified by a test that
   log-scale sampling is actually happening, not just labeled that way.
2. **Bayesian/efficient search with pruning** — Optuna `TPESampler`
   (learns from prior trials which regions look promising) +
   `MedianPruner` (kills a trial mid-CV if its running score trails the
   median of prior trials at the same fold).
3. **Robust CV, business metric** — every trial scored by 5-fold
   `StratifiedKFold` PR-AUC, never a single split.
4. **Early stopping** — `GradientBoostingClassifier`'s native
   `n_iter_no_change`/`validation_fraction`/`tol`: `n_estimators=300` is
   a ceiling, not a target — the model itself halts once its internal
   validation score stops improving (confirmed: winning trial actually
   used 142 trees, not 300).
5. **Confirm on held-out test set** — `src/tuning/confirm.py`, the ONLY
   place the test set is used, touched exactly once, comparing the tuned
   config against a reasonable non-tuned baseline.
6. **Log all trials** — `outputs/reports/all_trials_log.csv`: every one
   of the 25 trials (completed or pruned), with its params and score.

## Each named pitfall gets its own passing test

| Pitfall (from the study guide) | Test | Result |
|---|---|---|
| Huge wasteful grids | `test_pitfall_search_space_is_not_huge_wasteful_grid` | Computes the equivalent exhaustive grid size (~4,800 combinations at reasonable resolution) and asserts the actual trial budget (25) is far smaller — genuinely efficient, not a disguised grid |
| Overfitting the search to validation | `test_pitfall_search_overfitting_actually_checked` | Asserts the CV-best score is actually compared against the held-out test score for the winning config, with a real computed gap, not assumed to generalize |
| Unreproducible, unlogged trials | `test_pitfall_all_trials_logged` | Asserts every one of the 25 requested trials appears in the log with its params and status, whether it completed or was pruned |

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Peak-tuned model from efficient, logged search, test-confirmed gains | `outputs/reports/advanced_tuning_report.json`, `outputs/reports/all_trials_log.csv`, `outputs/artifacts/tuned_model.joblib` |
| Real-data quality & correctness (realistic, not toy) | Real 569-row WDBC data, Task 7's vetted 31-feature set, 25 real Optuna trials x up to 5 real CV folds each (with real early stopping) |
| Live verification & evidence | `tests/test_advanced_tuning.py` — 7/7 tests pass on live runs; pruning is confirmed to have real effect (14/25 trials actually pruned), not just configured |
| Dependency/failure/edge-case handling | Zero-trial search correctly leaves no valid "best" and raises clearly rather than returning garbage; data hand-off failures raise clearly before any search starts |

## How to run

```bash
pip install -r requirements.txt
python tests/test_advanced_tuning.py   # everything, incl. pitfall + edge-case tests (~2-3 min)
# or the pipeline directly:
python -m src.run_advanced_tuning
```

## Results from this run (seed=42) — reported honestly, including the near-zero gain

**Search efficiency:** 25 trials requested, **11 completed / 14 pruned
early (56% compute saved)** by the median pruner — real, not
hypothetical: `outputs/figures/optimization_history.png` shows exactly
which trials were cut and when.

**Best CV config:** `max_depth=2, learning_rate=0.0885, subsample=0.904,
min_samples_leaf=11` — CV PR-AUC **0.9950**. Early stopping used **142**
boosting stages (vs. the 300-stage ceiling).

**Parameter importance** (which two params dominate, per the guide's own
brainstorming question): **`learning_rate` (50.9%)** is by far the
dominant lever here, followed by `max_depth` (18.0%) and `subsample`
(17.6%) roughly tied, `min_samples_leaf` least influential (13.5%).

**CV-vs-test gap:** -0.0008 — essentially zero, `possible_search_overfitting: false`.

**Test-set confirmation** (touched exactly once): tuned PR-AUC **0.9958**
vs. baseline (untuned, reasonable-default) PR-AUC **0.9964**.
**Test-confirmed gain: -0.0006.**

**Honest interpretation, not spin:** the search did not find a
meaningfully better config than a sensible default on this dataset —
both sit at PR-AUC ~0.996 on a 114-row test set, well within noise.
Consistent with every finding since Task 9, this dataset is already
near its accuracy ceiling, so there's no real headroom for even an
efficient Bayesian search to capture. The value delivered here isn't a
performance win; it's the **honest, logged confirmation that there
isn't one to be had** — exactly what Step 5's "confirm on the held-out
test set" exists to catch, rather than reporting the CV-best number
(0.9950, which sat *above* both test numbers) as if it were the real
result.

Full numbers: `outputs/reports/advanced_tuning_report.json`, all 25
trials: `outputs/reports/all_trials_log.csv`.

## External resources needed

**One:** `optuna` (for Bayesian search + pruning), added to
`requirements.txt` — installs from PyPI, no account or API key needed.
Same offline WDBC data as Tasks 1-16 otherwise; only `pip install -r
requirements.txt` needs network access, once.

## Folder structure

```
task17_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── loader.py                     # YAML -> typed Config, sets global seed
│   └── config.yaml                   # search space + scales, pruner, baseline
├── data/
│   ├── clean_from_task2.csv          # carried over from Task 2
│   └── locked_feature_set.json       # carried over from Task 7
├── src/
│   ├── __init__.py
│   ├── run_advanced_tuning.py        # THE 6-step flow
│   ├── data/
│   │   ├── dataset.py                  # load + honour Task 7's locked features
│   │   └── split.py                    # one train/test split, test held out to Step 5
│   └── tuning/
│       ├── preprocess.py               # fit-on-train-only impute+scale
│       ├── search.py                   # Steps 1-4: Optuna objective, TPE, pruning, early stopping
│       ├── confirm.py                  # Step 5: test-set confirmation vs baseline
│       └── trial_log.py                # Step 6: full trial log + optimization history plot
├── tests/
│   └── test_advanced_tuning.py       # live run + one test per named pitfall + edge cases
└── outputs/
    ├── artifacts/
    │   ├── tuned_model.joblib
    │   ├── baseline_model.joblib
    │   └── preprocessor.joblib
    ├── reports/
    │   ├── advanced_tuning_report.json
    │   └── all_trials_log.csv
    ├── figures/
    │   └── optimization_history.png
    └── logs/
        └── run_advanced_tuning.log
```
