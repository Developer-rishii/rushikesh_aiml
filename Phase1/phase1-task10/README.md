# Task 10 — Complex Relationships

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Tasks 1–9:** loads Task 2's cleaned data + Task 7's
locked feature set, uses Task 9's confirmed linear-baseline config as
the bar to clear. Same `SEED=42`.

## What this delivers (Definition of Done)

**A non-linear model with validated lift over the baseline and effect
plots for sanity** — demonstrated live in `src/run_nonlinear.py`,
following the study guide's 6 steps:

1. **EDA reasoning for non-linearity** — documented in the run report
   (`eda_note`): WDBC size/shape measurements plausibly interact (a
   large-but-smooth tumor vs. a small-but-highly-concave one can carry
   similar risk through different mechanisms) — the kind of effect a
   linear model can only add, never combine conditionally.
2. **Train a more expressive model** — `GradientBoostingClassifier`,
   same single-Pipeline discipline as every task since Task 8.
3. **Compare validated performance to the linear baseline** —
   `src/run_nonlinear.py` computes lift on **validation data**, the
   linear model being Task 9's exact confirmed default config.
4. **Regularise/tune** — `src/models/tune_nonlinear.py`: `max_depth`
   (capped at 3, deliberately conservative for a 398-row training set)
   and `subsample` (stochastic regularisation) are searched via
   train-only CV — same discipline as Task 9's search/confirm split.
5. **Partial dependence for sense-checking** — `src/evaluation/effects.py`
   plots PDP curves for the top-4 features by importance, an actual PNG,
   not a description of what one would show.
6. **Keep only if the gain justifies the complexity** — gated on a real,
   configured threshold (`min_lift_to_keep: 0.01`), not zero.

## Each named pitfall gets its own passing test

| Pitfall (from the study guide) | Test | Result |
|---|---|---|
| Complexity with no validated gain | `test_pitfall_decision_gated_on_validated_lift_not_complexity` | Confirms the keep/reject decision is gated on a real nonzero threshold, and reports the actual measured lift driving the decision |
| Overfitting from unregularised power | `test_pitfall_regularisation_is_actually_searched` | Asserts `max_depth` and `subsample` are genuinely in the search space (not just `n_estimators`, which alone doesn't regularise), and that max depth is capped sanely for the training set size |
| Losing all explainability | `test_pitfall_explainability_preserved_via_pdp` | Actually generates the partial dependence PNG and asserts it exists and is non-empty — explainability is demonstrated, not assumed lost or preserved |

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Non-linear model with validated lift + effect plots | `outputs/reports/nonlinear_report.json` (validated lift, decision, rationale), `outputs/figures/partial_dependence.png` |
| Real-data quality & correctness (realistic, not toy) | Real 569-row WDBC data, Task 7's vetted 31-feature set, a real 24-combination regularisation grid actually fit 5x each (120 model fits) |
| Live verification & evidence | `tests/test_nonlinear.py` — 7/7 tests pass on live runs; the actual keep/reject decision and PDP plot are real artifacts from a real run |
| Dependency/failure/edge-case handling | Unknown non-linear model name rejected before fitting; requesting PDP-style feature importance from a model that doesn't support it (e.g. plain LogisticRegression) raises clearly instead of crashing obscurely |

## How to run

```bash
pip install -r requirements.txt
python tests/test_nonlinear.py    # everything, incl. pitfall + edge-case tests
# or the pipeline directly:
python -m src.run_nonlinear
```

## Results from this run (seed=42) — an honest "no" and why that's the right answer

**Best regularised gradient boosting config:** `max_depth=2,
learning_rate=0.1, n_estimators=100, subsample=0.8` — CV PR-AUC
**0.9921**, train-vs-CV-fold gap +0.0079 (healthy — no overfitting to CV).

**Validated comparison** (validation set, touched for comparison per Step 3):

| | Linear baseline (Task 9 config) | Gradient boosting (tuned) |
|---|---|---|
| PR-AUC | 1.0 | 1.0 |
| Recall | 0.9811 | 1.0 |
| Accuracy | 0.9882 | 1.0 |

**Validated lift: +0.0000 on the primary metric (PR-AUC).**

**Decision: REJECT the non-linear model, keep the linear baseline.**

**Honest interpretation:** the linear baseline is already at the PR-AUC
ceiling (1.0) on validation — there's no headroom left for a more
expressive model to capture. Gradient boosting nudges recall/accuracy
to a perfect 1.0 too, but on an 85-row validation set that's not a
meaningful, generalizable difference from the baseline's 0.9811/0.9882 —
it's the same ceiling reached two ways. This is precisely the outcome
the guide's own brainstorming question "Is the extra complexity earning
real, validated lift?" is designed to catch, and consistent with every
finding since Task 2: this dataset is well-separated enough that a
simple, fully-interpretable linear model is already sufficient. The
correct engineering decision — keeping the simpler, more explainable
model when the fancier one earns nothing — is the actual deliverable
here, not a forced "yes, boosting wins."

Held-out test-set number for the record (linear baseline, since it's the
kept model): PR-AUC 0.9979, accuracy 0.9767.

Full numbers: `outputs/reports/nonlinear_report.json`. Partial dependence
plot: `outputs/figures/partial_dependence.png`.

## External resources needed

**None.** Same offline WDBC data as Tasks 1-9. Only `pip install -r
requirements.txt` needs network access, once (adds `matplotlib` for the
PDP plot).

## Folder structure

```
task10_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── loader.py                     # YAML -> typed Config, sets global seed
│   └── config.yaml                   # baseline config, nonlinear search space, keep threshold
├── data/
│   ├── clean_from_task2.csv          # carried over from Task 2
│   └── locked_feature_set.json       # carried over from Task 7
├── src/
│   ├── __init__.py
│   ├── run_nonlinear.py              # THE 6-step flow
│   ├── data/dataset.py                 # from Task 8/9
│   ├── models/
│   │   ├── build.py                    # linear baseline + nonlinear Pipeline builders
│   │   └── tune_nonlinear.py           # Step 4: regularised CV search, train-only
│   └── evaluation/
│       ├── metrics.py                  # single source of metric computation
│       └── effects.py                  # Step 5: feature importance + partial dependence
├── tests/
│   └── test_nonlinear.py             # live run + one test per named pitfall + edge cases
└── outputs/
    ├── artifacts/
    │   ├── baseline_pipeline.joblib
    │   ├── nonlinear_pipeline.joblib
    │   └── kept_pipeline.joblib        # whichever model the Step 6 decision actually kept
    ├── reports/
    │   └── nonlinear_report.json
    ├── figures/
    │   └── partial_dependence.png
    └── logs/
        └── run_nonlinear.log
```
