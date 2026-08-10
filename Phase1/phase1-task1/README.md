# Task 1 — System Ingestion & Model Environment Setup

PlaceMux · AI/ML Developer · Phase 1 · Day 1

## What this delivers (maps to the study guide's Definition of Done)
- Reproducible ML env — pinned deps + fixed `SEED=42` (`configs/config.py`)
- Correct, stratified, leak-free train/val/test split (70/15/15)
- Working smoke-test run (dummy baseline) with metrics logged to `experiments/experiment_log.csv`
- Notebook -> module discipline: all real logic lives in `src/`, the notebook only calls it
- Tests proving no leakage and full determinism (`tests/test_pipeline.py`)

## Dataset
Wisconsin Diagnostic Breast Cancer (569 rows, 30 numeric features, binary
target). This is a **real, measured clinical dataset**, not a synthetic
toy — but it ships inside scikit-learn, so ingestion works fully offline
with **no external downloads or API keys required**.

> If you'd rather use a different/larger real dataset (e.g. a Kaggle CSV),
> swap the body of `src/data_ingestion.py::load_raw_dataframe()` — every
> other stage (split, tracker, smoke test, pipeline) is dataset-agnostic
> and needs zero changes.

## Folder structure
```
placemux-task01/
├── configs/
│   └── config.py            # single source of truth: SEED, paths, split ratios
├── data/                     # generated at runtime (raw.csv, train/val/test.csv)
├── experiments/
│   ├── experiment_log.csv    # generated: append-only run log (params + metrics)
│   └── models/                # generated: saved model artifacts (.joblib)
├── notebooks/
│   └── starter_notebook.ipynb
├── src/
│   ├── data_ingestion.py     # load + verify shape/types/class balance
│   ├── data_split.py         # stratified train/val/test, leakage guard
│   ├── experiment_tracker.py # CSV run logger
│   ├── smoke_test.py         # dummy baseline, fit/predict/log end-to-end
│   └── run_pipeline.py       # orchestrates all stages, fails loudly by stage
├── tests/
│   └── test_pipeline.py      # reproducibility + leakage + edge-case tests
├── run_log.txt                # captured stdout of a real, successful run
├── requirements.txt
└── README.md
```

## How to run (fresh machine)
```bash
python -m venv .venv && source .venv/bin/activate      # or: conda create -n placemux python=3.12
pip install -r requirements.txt

python -m src.run_pipeline        # runs ingest -> split -> smoke test end-to-end
python -m pytest tests/ -v        # 4/4 tests should pass
```

## Evidence this was actually run (not just claimed)
- `run_log.txt` — real stdout from `python -m src.run_pipeline`:
  shape `(569, 31)`, class balance `{1: 0.627, 0: 0.373}`, split
  `397/86/86`, baseline `val_accuracy=0.6279`.
- `experiments/experiment_log.csv` — the logged run row.
- `experiments/models/dummy_baseline.joblib` — the saved model artifact.
- All 4 tests in `tests/test_pipeline.py` pass.

## Pitfalls explicitly guarded against (per study guide §11)
| Pitfall | Guard |
|---|---|
| No fixed seeds | `SEED=42` centralized in `configs/config.py`, used everywhere |
| Leaking test data into training | index-overlap assertion in `data_split.py` + a dedicated test |
| Living in one giant notebook | all logic in `src/`; notebook only imports and calls it |

## Error / edge-case handling
- `data_ingestion.py`: raises on empty data, missing target column, non-numeric
  features, suspiciously small row count; warns on missing values / severe imbalance.
- `data_split.py`: raises if split fractions don't sum to 1.0; raises on any
  index overlap between splits (leakage).
- `smoke_test.py`: raises `FileNotFoundError` with a clear message if splits
  weren't generated first (explicit hand-off contract between stages).
- `run_pipeline.py`: wraps each stage so a failure names the exact stage.

## External resources needed
**None** for the dataset (bundled with scikit-learn — fully offline). You
only need internet access once, to `pip install -r requirements.txt`
(or use a machine that already has numpy/pandas/scikit-learn/joblib
installed). No API keys, no GPU, no MLflow server required — MLflow was
listed as optional in the study guide; a CSV-based tracker
(`experiment_tracker.py`) satisfies the "even a CSV/MLflow" requirement.
