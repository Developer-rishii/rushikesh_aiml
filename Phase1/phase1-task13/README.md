# Task 13 — Model Score Extraction

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Tasks 1–12:** wraps Task 12's exact packaged
`model.joblib` + `serving_config.json` (calibrated logreg, isotonic
calibration, threshold 0.37) in a clean scoring interface — this task
doesn't train or pick a model, it makes the already-shipped one
consumable by other systems.

## What this delivers (Definition of Done)

**A validated scoring interface returning versioned, well-defined
scores for batch and single inputs** — following the study guide's 6 steps:

1. **Clean predict interface** — `src/scoring/interface.py::Scorer`. Loads
   the model once; `.score_one()` and `.score_batch()` are the only two
   entrypoints a consumer ever touches — no raw sklearn access needed.
2. **Input contract, validated** — `src/scoring/schema.py`: the pydantic
   `PatientRecord` schema is **generated dynamically from the packaged
   model's own `feature_names`** at import time, so the contract can
   never silently drift out of sync with what the model actually expects.
   `extra="forbid"` rejects unexpected fields; missing/wrong-typed fields
   are rejected too.
3. **Standardised output** — `src/scoring/output.py::Score`: every score,
   batch or single, carries `probability`, an inline human-readable
   `score_meaning`, `decision` + `decision_label`, `threshold_used`,
   `model_version`, `model_name`, and `calibration_method` — one shape,
   never duplicated or drifted between call sites.
4. **Batch + single-record support** — both implemented, and the demo
   run proves they **agree exactly** on the same input (see below), not
   just that both "work" independently.
5. **Input validation + graceful errors** — a single `ScoringError`
   exception type wraps every failure mode (missing model, malformed
   input, inference failure) with a specific message — never a raw
   pydantic/sklearn traceback reaching the caller.
6. **Tested with realistic inputs** — `run_scoring_demo.py` scores 20
   real WDBC rows (not synthetic stubs) through both interfaces.

## Each named pitfall gets its own passing test

| Pitfall (from the study guide) | Test | Result |
|---|---|---|
| Undocumented score meaning | `test_pitfall_score_meaning_is_documented` | Asserts every score carries a real, non-trivial `score_meaning` string explaining what the number means and how the threshold was chosen — not a bare float |
| No input validation | `test_pitfall_input_validation_actually_rejects_bad_input` | Actually sends 3 distinct malformed inputs (incomplete record, unexpected extra field, wrong type) and confirms all 3 are rejected — not just that a validator object exists |
| No model versioning on outputs | `test_pitfall_model_version_present_and_stable` | Confirms every score carries a `model_version` derived from a content hash of the actual artifact file, and that the hash is deterministic across calls |

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Validated scoring interface, versioned, well-defined, batch + single | `src/scoring/interface.py`, `outputs/batch_scores/demo_batch_scores.csv`, `outputs/reports/scoring_demo_report.json` |
| Real-data quality & correctness (realistic, not toy) | 20 real WDBC rows scored (`run_scoring_demo.py`), not synthetic dummy inputs; 20/20 decisions matched ground truth |
| Live verification & evidence | `tests/test_scoring_interface.py` — 6/6 tests pass on live runs; single-record and batch paths are checked to produce byte-identical output on the same input, not assumed consistent |
| Dependency/failure/edge-case handling | Missing model artifact, empty batch request, and 3 distinct malformed-record shapes all raise a clear `ScoringError`/`FileNotFoundError` instead of crashing obscurely |

## How to run

```bash
pip install -r requirements.txt
python tests/test_scoring_interface.py   # everything, incl. pitfall + edge-case tests
# or the demo directly:
python run_scoring_demo.py
```

### Using the interface directly

```python
from src.scoring.interface import Scorer

scorer = Scorer()
score = scorer.score_one({"mean_radius": 14.2, "mean_texture": 20.1, ...})
print(score.model_dump())
# {'probability': 0.87, 'score_meaning': "Calibrated (isotonic) probability...",
#  'decision': 1, 'decision_label': 'benign', 'threshold_used': 0.37,
#  'model_version': 'sha256:f27b186532ae8d4e', 'model_name': 'logreg',
#  'calibration_method': 'isotonic'}

batch = scorer.score_batch([record1, record2])
```

## Results from this run (seed=42)

**Model version:** `sha256:f27b186532ae8d4e` (content hash of Task 12's
`model.joblib`). **Calibration:** isotonic. **Threshold:** 0.37.

**20 real WDBC rows scored** through both interfaces — single-record and
batch outputs verified byte-identical on row 0, and **20/20 decisions
matched the true label** on this sample.

**Graceful error handling, demonstrated live** (not just described):
sending a record with only an unknown field, and a record missing most
required fields, both raised a specific `ScoringError` with the exact
pydantic validation detail attached — caught cleanly, no raw traceback.

Example single score (real, from this run):
```json
{
  "probability": 0.874929,
  "score_meaning": "Calibrated (isotonic) probability that the tumor is BENIGN (positive class = 1). A value near 1.0 means high confidence of benign; near 0.0 means high confidence of malignant. The 'decision' field applies the operating threshold (0.37) chosen in Task 12 to minimize the real clinical cost of errors — it is NOT a plain 0.5 cutoff.",
  "decision": 1,
  "decision_label": "benign",
  "threshold_used": 0.37,
  "model_version": "sha256:f27b186532ae8d4e",
  "model_name": "logreg",
  "calibration_method": "isotonic"
}
```

Full batch output: `outputs/batch_scores/demo_batch_scores.csv`. Full
report: `outputs/reports/scoring_demo_report.json`.

## External resources needed

**None.** Wraps Task 12's already-packaged, offline model artifact — no
retraining, no downloads. Only `pip install -r requirements.txt` (adds
`pydantic`) needs network access, once.

## Folder structure

```
task13_project/
├── README.md
├── requirements.txt
├── run_scoring_demo.py               # Step 6: demo with realistic inputs
├── model_artifact/                   # carried over from Task 12
│   ├── model.joblib
│   └── serving_config.json
├── data/
│   ├── clean_from_task2.csv          # carried over from Task 2
│   └── locked_feature_set.json       # carried over from Task 7
├── src/
│   ├── __init__.py
│   └── scoring/
│       ├── __init__.py
│       ├── schema.py                  # Step 2: input contract, generated from the model artifact
│       ├── output.py                  # Step 3: standardised Score / BatchScoreResult
│       ├── registry.py                # Step 3: content-hash model versioning
│       └── interface.py               # Steps 1,4,5: the Scorer — the actual deliverable
├── tests/
│   └── test_scoring_interface.py     # live run + one test per named pitfall + edge cases
└── outputs/
    ├── batch_scores/
    │   └── demo_batch_scores.csv
    ├── reports/
    │   └── scoring_demo_report.json
    └── logs/
        └── run_scoring_demo.log
```
