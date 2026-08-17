# Task 6 — The Binary Decision

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Tasks 1–5:** reuses Task 4's preprocessing pipeline,
Task 2's cleaned WDBC dataset, and Task 5's model registry. Same
`SEED=42` throughout.

## What this delivers (Definition of Done)

**A binary classifier reported with confusion matrix, precision/recall
and a justified threshold** — demonstrated live in
`src/run_binary_decision.py`, following the study guide's 6 steps:

1. **Train** — `LogisticRegression` on Task 4's preprocessed features.
2. **Confusion matrix on validation** — `src/evaluation/decision_metrics.py`,
   with cells named by clinical meaning (`missed_malignancy`,
   `unnecessary_biopsy`), not just TP/FP/TN/FN, because sklearn's
   "positive class" convention (class 1 = benign) makes raw FP/FN labels
   easy to misread for this dataset — see the module docstring.
3. **Precision, recall, F1** — reported together, never accuracy alone.
4. **ROC/PR curves + threshold pick** — `src/evaluation/curves.py` plots
   both across all thresholds; `src/evaluation/threshold_selection.py`
   sweeps a threshold grid and picks the one minimizing **expected cost**.
5. **Imbalance check** — `src/evaluation/imbalance_check.py` computes the
   majority-baseline accuracy directly and flags when accuracy would mislead.
6. **Cost-tied threshold recommendation** — documented reasoning: a missed
   malignancy costs 10x an unnecessary biopsy (config `cost:` section),
   so the threshold is chosen to minimize `missed_malignancy*10 +
   unnecessary_biopsy*1`, not left at the arbitrary 0.5 default.

## Each named pitfall gets its own passing test

| Pitfall (from the study guide) | Test | Result |
|---|---|---|
| Accuracy-only reporting | `test_pitfall_not_accuracy_only` | Asserts precision/recall/F1 are all reported alongside accuracy, not accuracy in isolation |
| Default threshold with no cost reasoning | `test_pitfall_threshold_is_cost_justified_not_default` | Asserts a documented cost rationale exists and the chosen threshold actually minimizes expected cost vs. the 0.5 default (not just claims to) |
| Ignoring imbalance entirely | `test_pitfall_imbalance_not_ignored` | Asserts the majority-baseline accuracy is explicitly computed and flagged as potentially misleading |

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Reproducible env, correct split, smoke-test w/ logged metrics | `SEED=42`; stratified 70/15/15 split + leakage guard (Task 4); `outputs/reports/binary_decision_report.json` + `outputs/logs/run_binary_decision.log` from a real run |
| Real-data quality & correctness (realistic, not toy) | Real 569-row WDBC data with Task 4's categorical + missing-value enrichment carried through |
| Live verification & evidence | `tests/test_binary_decision.py` — 7/7 tests pass on live runs; actual ROC/PR PNG plots and a full threshold sweep CSV are produced, not described |
| Dependency/failure/edge-case handling | Data/model/metrics stages wrapped with specific errors + `sys.exit(1)`; tests cover unknown metric name and threshold values at the extremes (0.0001, 0.9999) |

## How to run

```bash
pip install -r requirements.txt
python tests/test_binary_decision.py   # everything, incl. pitfall tests + edge cases
# or just the pipeline:
python -m src.run_binary_decision
```

## Results from this run (seed=42)

**Confusion matrix @ default threshold 0.5** (validation, n=85):

| | Predicted malignant | Predicted benign |
|---|---|---|
| **Actually malignant** | 32 (correct) | 0 (missed_malignancy) |
| **Actually benign** | 2 (unnecessary_biopsy) | 51 (correct) |

Precision 1.0, recall 0.9623, F1 0.9808, accuracy 0.9765, ROC-AUC 1.0, PR-AUC 1.0.

**Imbalance check:** validation set is 62.4%/37.7% — a majority-class
baseline would score 62.4% accuracy on nothing meaningful, flagged in the
report as `accuracy_is_potentially_misleading: true`.

**Cost-based threshold recommendation:** with `missed_malignancy` costed
10x `unnecessary_biopsy`, the sweep finds **threshold 0.32** achieves
**zero total expected cost** (0 missed malignancies, 0 unnecessary
biopsies) — an improvement over the already-good default-0.5 result (2
unnecessary biopsies, cost 2.0). Honest note: this dataset is well-
separated (consistent with Tasks 2–5's findings), so both thresholds
already avoid any missed malignancy — the sweep's value here is
*confirming* there's headroom to also cut unnecessary biopsies to zero,
not fixing a dangerous default.

Full sweep: `outputs/reports/threshold_sweep.csv`. Full report:
`outputs/reports/binary_decision_report.json`. Curve plots:
`outputs/figures/roc_curve.png`, `outputs/figures/pr_curve.png`.

## External resources needed

**None.** Same offline WDBC data as Tasks 1–5. Only `pip install -r
requirements.txt` needs network access, once (adds `matplotlib` for the
ROC/PR plots).

## Folder structure

```
task6_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── loader.py                     # YAML -> typed Config, sets global seed
│   └── config.yaml                   # paths, model, cost-per-error-type
├── data/
│   └── clean_from_task2.csv          # carried over from Task 2
├── src/
│   ├── __init__.py
│   ├── run_binary_decision.py        # THE 6-step flow
│   ├── data/dataset.py                 # from Task 4
│   ├── preprocessing/pipeline.py       # from Task 4
│   ├── modeling/registry.py            # from Task 5
│   └── evaluation/
│       ├── decision_metrics.py         # Step 2/3: confusion matrix, precision/recall/F1
│       ├── curves.py                   # Step 4: ROC/PR curve data + plots
│       ├── threshold_selection.py      # Step 4/6: cost-based threshold sweep
│       └── imbalance_check.py          # Step 5: imbalance flagging
├── tests/
│   └── test_binary_decision.py        # live run + one test per named pitfall + edge cases
└── outputs/
    ├── models/
    │   ├── fitted_preprocessor.joblib
    │   └── logreg.joblib
    ├── reports/
    │   ├── binary_decision_report.json
    │   └── threshold_sweep.csv
    ├── figures/
    │   ├── roc_curve.png
    │   └── pr_curve.png
    └── logs/
        └── run_binary_decision.log
```
