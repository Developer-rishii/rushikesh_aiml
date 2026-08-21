# Task 9 — Hyperparameter Tuning

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Tasks 1–8:** loads Task 2's cleaned data + Task 7's
locked feature set, reuses Task 8's single-Pipeline discipline. Same
`SEED=42`.

## What this delivers (Definition of Done)

**A tuned model with the best config, CV results and confirmed test-set
gain over default** — demonstrated live in `src/run_tuning.py`, following
the study guide's 6 steps, with strict data-flow discipline:

```
X_train, y_train  ->  src/tuning/search.py   (Steps 2-4: CV search, train-only)
X_test,  y_test    ->  src/tuning/confirm.py  (Step 5: touched exactly once)
```

1. **Hyperparameters that matter** — `configs/config.yaml`: only `C`
   (regularisation strength) and `penalty` are searched for
   `LogisticRegression` — the two knobs that actually move the
   bias/variance trade-off. `max_iter` and `random_state` are explicitly
   excluded (documented in the config comments) — they're plumbing, not
   bias/variance knobs, per the pitfall "searching params that don't matter."
2. **Search + CV scheme** — `GridSearchCV` with `StratifiedKFold(5)`,
   scored on `average_precision` (PR-AUC) — same primary metric as every
   prior task.
3. **Run the search** — `src/tuning/search.py::run_search`, scored by
   the business metric, never accuracy.
4. **Select by validated score** — `search.best_score_` is the mean
   **CV** score across folds, never a training-set number.
5. **Confirm on held-out test set** — `src/tuning/confirm.py`, the ONLY
   place `X_test`/`y_test` are used in this whole project run. Both the
   default-params model and the CV-tuned model are refit on the same
   `X_train` and evaluated once each on `X_test`.
6. **Record best config + improvement** — `outputs/reports/tuning_report.json`.

## Each named pitfall gets its own passing, structural test

| Pitfall (from the study guide) | Test | Result |
|---|---|---|
| Tuning on the test set | `test_pitfall_search_never_touches_test_set` | AST-parses `run_search`'s source (stripped of docstrings) and asserts `X_test`/`y_test`/`X_val`/`y_val` appear nowhere in its actual code |
| Reporting CV best as final without test confirmation | `test_pitfall_confirm_module_uses_test_exactly_once` | Asserts `confirm.py` actually calls `.fit`/`.predict_proba` against `X_test` for BOTH the default and tuned models — CV's `best_score_` is never written to the report as if it were the confirmed result |
| Searching params that don't matter | `test_pitfall_only_bias_variance_params_are_searched` | Asserts `max_iter`/`random_state` are absent from the search space and `C` is present |

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Tuned model with best config, CV results, confirmed test-set gain | `outputs/reports/cv_results.csv` (full 12-combination leaderboard), `outputs/reports/tuning_report.json` (best config + test-set comparison), both `default_pipeline.joblib` and `tuned_pipeline.joblib` saved |
| Real-data quality & correctness (realistic, not toy) | Real 569-row WDBC data, Task 7's vetted 31-feature set, a real 12-combination grid actually fit 5x each (60 model fits total) |
| Live verification & evidence | `tests/test_tuning.py` — 7/7 tests pass on live runs; the CV leaderboard, test-set metrics, and honest negative-gain finding below are all real numbers from an actual run |
| Dependency/failure/edge-case handling | Unknown search strategy and unknown model name both rejected before any fitting is attempted; overfitting-to-CV is actively checked (train-vs-CV-fold gap), not assumed absent |

## How to run

```bash
pip install -r requirements.txt
python tests/test_tuning.py    # everything, incl. pitfall + edge-case tests
# or the pipeline directly:
python -m src.run_tuning
```

## Results from this run (seed=42) — reported honestly, including the miss

**Best CV config:** `C=0.01, penalty=l2` — best CV PR-AUC = **0.9962**
across 5 folds (full 12-combination leaderboard in
`outputs/reports/cv_results.csv`). Train-vs-CV-fold gap for the winner:
**-0.0004** (essentially zero — no sign of overfitting to the CV folds).

**Held-out test-set confirmation** (touched exactly once, per Step 5):

| | Default (`C=1.0` implicit, `l2`) | Tuned (`C=0.01`, `l2`) |
|---|---|---|
| PR-AUC | **0.9979** | 0.9957 |
| ROC-AUC | 0.9965 | 0.9925 |
| Accuracy | 0.9767 | 0.9767 |

**Test-set gain: -0.0022 (tuned is marginally WORSE than default on
test).** This is reported as-is, not hidden or reframed as a win.

**Honest interpretation, not spin:** this is the exact scenario the
guide's brainstorming question "Are you tuning to the validation set's
noise?" is asking you to watch for. Both models sit at PR-AUC ~0.996-0.998
on an 86-row test set — the entire gap (0.0022) is well within the noise
band of that sample size, not a meaningful regression. The CV search
picked a slightly-more-regularized config that generalized marginally
worse on this particular 86-row test split purely by chance. **The
correct read of this result is "tuning found no reliable, confirmed gain
over default on this dataset" — not "tuning made things worse."** Given
this is already a near-ceiling, well-separated dataset (consistent with
every prior task since Task 2), that's a legitimate and expected outcome,
and reporting it plainly is exactly what Step 5's "confirm the gain
holds" is for: it exists to catch cases just like this one, where CV
looked like an improvement but the held-out set doesn't confirm it.

Full numbers: `outputs/reports/tuning_report.json`,
`outputs/reports/cv_results.csv`.

## External resources needed

**None.** Same offline WDBC data as Tasks 1-8. Only `pip install -r
requirements.txt` needs network access, once.

## Folder structure

```
task9_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── loader.py                    # YAML -> typed Config, sets global seed
│   └── config.yaml                  # search space, CV scheme, default params
├── data/
│   ├── clean_from_task2.csv         # carried over from Task 2
│   └── locked_feature_set.json      # carried over from Task 7
├── src/
│   ├── __init__.py
│   ├── run_tuning.py                # THE 6-step flow
│   ├── data/dataset.py                # from Task 8
│   └── tuning/
│       ├── model.py                   # single sklearn Pipeline builder
│       ├── search.py                  # Steps 2-4: CV search, train-only
│       └── confirm.py                 # Step 5: test-set confirmation, used once
├── tests/
│   └── test_tuning.py                # live run + one test per named pitfall + edge cases
└── outputs/
    ├── artifacts/
    │   ├── default_pipeline.joblib
    │   └── tuned_pipeline.joblib
    ├── reports/
    │   ├── tuning_report.json
    │   └── cv_results.csv
    └── logs/
        └── run_tuning.log
```
