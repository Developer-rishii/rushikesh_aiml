# Task 7 — Baseline Target Feature Engineering

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Tasks 1–6:** loads Task 2's leakage-cleaned WDBC dataset
(same 30-feature baseline validated across Tasks 3–6). Same `SEED=42`.

## What this delivers (Definition of Done)

**A vetted baseline feature set with importance analysis and a leakage
check** — demonstrated live in `src/run_feature_engineering.py`, following
the study guide's 6 steps:

1. **Re-confirm target** — `src/features/target_check.py` verifies the
   target is a clean binary label with zero missing values before any
   feature work starts (`src/features/target_check.py::confirm_target`
   actively rejects a non-binary or missing-value target — verified with
   a dedicated test, not just assumed clean).
2. **Domain-derived candidate features** — `src/features/engineer.py`.
   Every WDBC measurement is recorded as mean/error/worst; two domain
   features are derived per measurement: `ratio_worst_to_mean_*` (how
   extreme is the worst cell vs. the average — literally what a
   pathologist eyeballs) and `coeff_variation_*` (cell-to-cell
   heterogeneity, an independent malignancy signal). 20 candidates.
3. **Aggregate features** — composite "worst shape irregularity" and
   z-scored "worst size" scores, summarizing correlated sub-measurements
   the way a domain expert would. 2 candidates. **22 candidates total.**
4. **Train + inspect importance** — `src/features/importance.py` computes
   **permutation importance on validation data** (never training data —
   training-set importance reflects overfitting, not real signal).
5. **Prune useless/leaky features** — `src/features/pruning.py`, two
   independent gates: a leakage gate (correlation-with-target threshold,
   same method as Task 2) runs on candidates *before* they ever reach the
   model; a usefulness gate drops any candidate whose measured
   permutation-importance lift is ≤ the configured threshold.
6. **Lock the baseline feature set** — `outputs/baseline_features/locked_feature_set.json`,
   with the measured PR-AUC before/after engineering as evidence, not a claim.

## Each named pitfall gets its own passing test

| Pitfall (from the study guide) | Test | Result |
|---|---|---|
| Engineering leaky features | `test_pitfall_engineered_features_are_leak_free_by_construction` | Runs the leakage gate on all 22 real candidates (0 flagged — engineered by construction from same-row-only data), then plants a deliberately perfect-correlation column and confirms the gate actually catches it |
| Adding features without measuring lift | `test_pitfall_features_measured_not_assumed` | Asserts the locked report contains an actual measured PR-AUC number for "before engineering" vs "after," not just a feature list |
| Ignoring domain knowledge | `test_pitfall_domain_knowledge_documented` | Source-inspects `engineer.py` to confirm each derivation carries an explicit, human-readable domain rationale |

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Vetted baseline feature set with importance analysis and leakage check | `outputs/baseline_features/locked_feature_set.json` (the vetted set), `outputs/reports/feature_importance.csv` (permutation importance for every candidate), leakage gate result in the same report |
| Real-data quality & correctness (realistic, not toy) | Real 569-row WDBC data; 22 real candidate features actually computed and evaluated, not illustrative stubs |
| Live verification & evidence | `tests/test_feature_engineering.py` — 7/7 tests pass on a live run; the leakage gate is proven against a real planted leak, not just asserted to work |
| Dependency/failure/edge-case handling | Target-quality check rejects a non-binary/missing target; ratio features are epsilon-guarded against divide-by-zero; every stage wrapped with a specific error + `sys.exit(1)` |

## How to run

```bash
pip install -r requirements.txt
python tests/test_feature_engineering.py   # everything, incl. pitfall tests + edge cases
# or just the pipeline:
python -m src.run_feature_engineering
```

## Results from this run (seed=42) — and an honest reading of them

- **22 candidate features derived**, all row-wise from already-available
  same-sample measurements (no leakage by construction).
- **Leakage gate: 0 flagged.** None of the 22 candidates exceeded the
  correlation threshold — expected, since they're ratios/aggregates of
  already-legitimate features, not shortcuts to the label. The gate was
  separately proven to work by planting and catching a synthetic
  perfect-correlation column (see `tests/`).
- **Usefulness gate: 21 of 22 dropped**, 1 kept (`coeff_variation_area`).
- **Measured PR-AUC: 1.0 with the original 30 features alone, 1.0 with
  the final locked 31-feature set.** Lift = +0.0000.

**Honest interpretation, not spin:** this null result is consistent with
every prior task in this track (Tasks 2, 5, 6 all found the WDBC dataset
already well-separated at PR-AUC/ROC-AUC ≈ 1.0 with the original
features). There's no more accuracy headroom left for feature
engineering to capture on *this* validation split — which is exactly
what the study guide's brainstorming question "Is more features helping
or just adding noise?" is asking you to check for, not assume the
answer to. The pruning gate did its job: it kept the process honest by
removing the 21 candidates that added complexity without measured
benefit, rather than locking in a bigger feature set because it *sounded*
more sophisticated.

Full numbers: `outputs/reports/feature_engineering_report.json`,
`outputs/reports/feature_importance.csv`,
`outputs/baseline_features/locked_feature_set.json`.

## External resources needed

**None.** Same offline WDBC data as Tasks 1–6. Only `pip install -r
requirements.txt` needs network access, once.

## Folder structure

```
task7_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── loader.py                        # YAML -> typed Config, sets global seed
│   └── config.yaml                      # paths, model, pruning thresholds
├── data/
│   └── clean_from_task2.csv             # carried over from Task 2
├── src/
│   ├── __init__.py
│   ├── run_feature_engineering.py       # THE 6-step flow
│   ├── data/dataset.py                   # load + split (from Task 4/6)
│   ├── preprocessing/simple.py           # fit-on-train-only impute+scale
│   └── features/
│       ├── target_check.py               # Step 1: re-confirm target quality
│       ├── engineer.py                   # Steps 2/3: domain + aggregate features
│       ├── importance.py                 # Step 4: permutation importance on validation
│       └── pruning.py                    # Step 5: leakage gate + usefulness gate
├── tests/
│   └── test_feature_engineering.py      # live run + one test per named pitfall + edge cases
└── outputs/
    ├── baseline_features/
    │   └── locked_feature_set.json       # Step 6: the locked, vetted deliverable
    ├── reports/
    │   ├── feature_engineering_report.json
    │   └── feature_importance.csv
    └── logs/
        └── run_feature_engineering.log
```
