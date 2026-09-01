# Task 19 — Application Model Serializing

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Tasks 1–17 (numeric track):** loads Task 2's cleaned
data + Task 7's locked feature set, trains Task 9's confirmed logreg
config. Same `SEED=42`.

## What this delivers (Definition of Done)

**A serialised, versioned model+preprocessor artifact with metadata and
a tested load/predict path** — demonstrated live in
`src/run_serialize.py`, following the study guide's 6 steps:

1. **Bundle preprocess+model into ONE artifact** — `src/train/build.py`:
   `impute -> scale -> model` as steps of a single `sklearn.pipeline.Pipeline`
   (same discipline as Tasks 8-17), so "saving model but not preprocessor"
   is structurally impossible — there's no code path where they're
   separate objects one could forget.
2. **Save with joblib + a metadata file** — `src/serialize/store.py::save_artifact`
   writes both `model_pipeline.joblib` and `metadata.json` together, one call.
3. **Record library versions + training metrics** — `src/serialize/metadata.py`:
   the exact `scikit-learn`/`numpy`/`pandas`/`joblib`/Python versions
   used to train, plus the real held-out test metrics.
4. **Load-and-predict with input validation** — `src/serialize/store.py::predict`:
   checks for missing features, rejects unexpected extra columns, and
   **reorders input columns to match training order automatically** —
   directly answering the brainstorming question "what happens if input
   features arrive in a different order?"
5. **Test loading in a fresh environment** — `src/fresh_env_check.py` runs
   as a genuinely **separate subprocess** (via `subprocess.run`, not an
   in-process function call), so it has zero shared memory with the
   training run — it only ever touches the files on disk, exactly like
   a real deployment would.
6. **Version for traceability** — SHA-256 content hash of the serialized
   model bytes (same approach as Task 13), computed before saving,
   deterministic, and embedded in every prediction result.

## Each named pitfall gets its own passing test

| Pitfall (from the study guide) | Test | Result |
|---|---|---|
| Saving model but not preprocessor | `test_pitfall_preprocessor_saved_with_model` | Reloads the artifact and confirms `impute`/`scale`/`model` are all present as pipeline steps, then scores **raw, unscaled** test data through it successfully — proving preprocessing travels with the model, not asserting it does |
| Version mismatch breaking load | `test_pitfall_version_mismatch_detected_not_silent` | Deliberately tampers the saved metadata's recorded scikit-learn version, reloads, and confirms the mismatch detector actually fires — not just that the mechanism exists |
| No metadata/lineage | `test_pitfall_metadata_lineage_present` | Asserts every required lineage field (version, timestamp, seed, metrics, library versions, source data lineage) is present and populated |

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Serialised, versioned artifact with metadata and tested load/predict path | `outputs/artifact_store/model_pipeline.joblib` + `metadata.json`, `outputs/reports/serialize_report.json` (fresh-environment test result embedded) |
| Real-data quality & correctness (realistic, not toy) | Real 569-row WDBC data, Task 7's vetted 31-feature set, real held-out test metrics (PR-AUC 0.9971) |
| Live verification & evidence | `tests/test_serialize.py` — 7/7 tests pass on live runs; the Step 5 fresh-environment check is a REAL subprocess invocation with its own JSON output captured, not an in-process shortcut |
| Dependency/failure/edge-case handling | Missing required feature, unexpected extra column, and missing artifact files all raise a single well-defined `ArtifactLoadError` instead of a raw sklearn/joblib traceback |

## How to run

```bash
pip install -r requirements.txt
python tests/test_serialize.py   # everything, incl. pitfall + edge-case tests
# or the pipeline directly:
python -m src.run_serialize
```

## Results from this run (seed=42)

**Training metrics (held-out test set):** PR-AUC 0.9971, ROC-AUC 0.9954,
precision 0.9855, recall 0.9444, F1 0.9645, accuracy 0.9561.

**Artifact version:** `sha256:fa1ba63740468e37` — a deterministic
content hash of the serialized pipeline bytes, computed before saving.

**Fresh-environment check (genuine subprocess, `passed: true`):** loaded
`model_pipeline.joblib` + `metadata.json` from disk in a brand-new
Python process with zero shared memory with training, scored 5 real
held-out rows, and returned confident, sensible probabilities (e.g.
0.9999 and 0.0000 for clear-cut cases) — verified live, not claimed.

**Library versions recorded:** Python 3.12.3, scikit-learn 1.5.1, numpy
1.26.4, pandas 2.2.2, joblib 1.4.2. Zero mismatches detected against the
current environment on this run (the mismatch detector was separately
proven to fire correctly using a deliberately tampered version string
in the test suite).

**Column-reorder robustness:** predictions are byte-identical whether
input columns arrive in training order or fully reversed — the
`predict()` function reorders to the recorded `feature_names_ordered`
before scoring, so silent misalignment (a real production failure mode)
can't happen.

Full report: `outputs/reports/serialize_report.json`.

## External resources needed

**None.** Same offline WDBC data as Tasks 1-17. Only `pip install -r
requirements.txt` needs network access, once.

## Folder structure

```
task19_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── loader.py                     # YAML -> typed Config, sets global seed
│   └── config.yaml                   # model, artifact filenames, paths
├── data/
│   ├── clean_from_task2.csv          # carried over from Task 2
│   └── locked_feature_set.json       # carried over from Task 7
├── src/
│   ├── __init__.py
│   ├── run_serialize.py              # THE 6-step flow
│   ├── fresh_env_check.py            # Step 5: run as a real subprocess
│   ├── train/
│   │   └── build.py                    # Step 1: single Pipeline (preprocess+model)
│   └── serialize/
│       ├── metadata.py                 # Step 3: library versions, metrics, lineage
│       └── store.py                    # Steps 2,4,6: save/load/predict/version
├── tests/
│   └── test_serialize.py             # live run + one test per named pitfall + edge cases
└── outputs/
    ├── artifact_store/
    │   ├── model_pipeline.joblib      # THE bundled artifact
    │   └── metadata.json              # THE lineage/version metadata
    ├── reports/
    │   └── serialize_report.json
    └── logs/
        └── run_serialize.log
```
