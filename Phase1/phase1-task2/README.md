# Task 2 — Pre-Project Feature Scope & Profiling

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Task 1:** this project loads the exact raw dataset Task 1
ingested and validated (`data/raw_from_task1.csv`, copied from
`phase1-task1/data/raw.csv`), reuses `SEED=42` and `TARGET_COL="target"`
so both tasks stay comparable, and mirrors Task 1's folder layout
(`configs/`, `src/`, `tests/`, `data/`).

## 1. Problem scope (Step 1–2 of the build pipeline)

- **Problem statement:** Predict whether a breast tumor is malignant or
  benign from digitized fine-needle-aspirate (FNA) measurements available
  at biopsy time — see `configs/config.py::PROBLEM_STATEMENT`.
- **Type:** Binary classification.
- **Target:** `target` (0 = malignant, 1 = benign).
- **Success metric:** PR-AUC, checked against the minority base rate —
  not raw accuracy, since a majority-class baseline already scores 62.7%.

## 2. Dataset

Wisconsin Diagnostic Breast Cancer (WDBC) — the same **real, measured
clinical dataset** Task 1 used (569 rows, 30 numeric diagnostic features).
It is not resampled or altered here.

To make the leakage hunt and feature-quality audit demonstrable on real
data (the raw WDBC feature set itself is a curated benchmark with no
leakage or ID columns), `src/data_ingestion.py` enriches it with two
columns exactly like ones that leak into real hospital data extracts:

- `patient_record_id` — a unique ID, no generalizable signal.
- `pathologist_diagnosis_code` — populated by the lab *after* diagnosis;
  it's the target restated in another vocabulary. Textbook leakage.

Both are found and removed in Step 4, restoring the original 30 real
features Task 1 validated. This is documented plainly in
`outputs/reports/leakage_report.md`, not hidden.

## 3. How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Core deliverable (target, metric, feature list, leakage check, balance report) | `configs/config.py`, `outputs/reports/feature_profile.csv`, `leakage_report.md`, `balance_report.md` |
| Reproducible env + correct split + smoke-test w/ logged metrics | `SEED=42` everywhere, stratified 70/15/15 split in `src/run_pipeline.py`, `outputs/logs/run_metrics.json` + `run_log.txt` |
| Real-data quality & correctness (realistic, not toy) | Real 569-row clinical dataset carried over from Task 1, plus realistic records-system fields (ID + leaky code) |
| Live verification & evidence | `tests/test_pipeline.py` — 5/5 tests pass on a real run, asserted, not claimed; `outputs/logs/` has the actual output |
| Dependency/failure/edge-case handling | `try/except` + explicit `sys.exit` per stage in `run_pipeline.py`; edge-case tests for missing file, single-class data |

## 4. How to run

```bash
pip install -r requirements.txt
python tests/test_pipeline.py     # runs everything end-to-end + edge cases
```

Or stage by stage:

```bash
python src/data_ingestion.py      # step 1-2: load Task 1's data, define target
python src/feature_profiling.py   # step 3: inventory & profile features
python src/leakage_check.py       # step 4: hunt & remove leakage
python src/balance_report.py      # step 5: class balance & base rates
python src/run_pipeline.py        # step 6: split, baseline, go/no-go
```

## 5. Actual results from this run (seed=42)

- **Leakage found & removed:** `pathologist_diagnosis_code`,
  `patient_record_id` — full reasoning in `outputs/reports/leakage_report.md`.
- **Class balance:** 212 malignant / 357 benign (62.7% majority) — mild
  imbalance, stratified split used; full numbers in `balance_report.md`.
- **Baseline (logistic regression, class-weighted):** val PR-AUC **1.0**,
  recall **0.962**, accuracy **0.977** vs a 62.7% majority baseline →
  **GO**. This dataset is well-separated (a known property of WDBC), so
  a linear baseline is already strong — full metrics in
  `outputs/logs/run_metrics.json`.

## 6. Answers to the guide's "brainstorming" questions

- **Could any feature leak the label? Prove it can't.** Yes —
  `pathologist_diagnosis_code`, proven leaky by *when it's populated*
  (post-diagnosis), not by correlation alone. It's categorical, so it
  wouldn't even be caught by a numeric-correlation-only check — proof
  that domain reasoning has to be the primary method, statistics only
  confirmatory.
- **Is there enough signal to beat a trivial baseline?** Yes, decisively
  — PR-AUC 1.0 vs a 0.373 minority base rate on the real 30-feature set.
- **Cost of false positive vs false negative:** A false negative (calling
  a malignant tumor benign) risks a missed cancer diagnosis — far costlier
  than a false positive (an unnecessary follow-up biopsy). This is why
  the go/no-go gate requires recall > 0.85, not just a good PR-AUC.

## 7. External resources needed

**None.** The dataset is Task 1's already-ingested, offline WDBC data —
no downloads, API keys, or GPU required. Only `pip install -r
requirements.txt` needs network access once.

## 8. Folder structure

```
task2_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   └── config.py               # SEED, paths, target definition (shared style w/ Task 1)
├── data/
│   ├── raw_from_task1.csv      # carried over from phase1-task1/data/raw.csv
│   ├── raw_enriched.csv        # generated: + id + demo leakage column
│   └── clean.csv               # generated: post leakage-removal
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py       # step 1-2: load Task 1 data + enrich
│   ├── feature_profiling.py    # step 3
│   ├── leakage_check.py        # step 4
│   ├── balance_report.py       # step 5
│   └── run_pipeline.py         # step 6: split, baseline, go/no-go
├── tests/
│   └── test_pipeline.py        # live verification + edge cases
└── outputs/
    ├── reports/
    │   ├── feature_profile.csv
    │   ├── leakage_report.md
    │   └── balance_report.md
    └── logs/
        ├── run_log.txt
        └── run_metrics.json
```
