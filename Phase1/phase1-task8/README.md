# Task 8 — The End-to-End Pipeline

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Tasks 1–7:** consumes Task 2's leakage-cleaned WDBC
data and Task 7's vetted, locked feature set (`data/locked_feature_set.json`)
directly — this task assembles the wiring, it doesn't re-derive features
or re-run leakage checks that were already done upstream. Same `SEED=42`.

## What this delivers (Definition of Done)

**A one-command, reproducible end-to-end pipeline that outputs a model
and its metrics** — `python run.py`, nothing else required, following
the study guide's 6 steps:

1. **Single sklearn Pipeline** — `src/pipeline/build.py::build_pipeline`
   chains `impute -> scale -> model` as ONE `sklearn.pipeline.Pipeline`
   object. There is no separate preprocessor object anywhere in this
   project — see the pitfall table below for how this is *proven*, not
   just structured to look that way.
2. **Data loading + splitting at the front** — `src/data/dataset.py`,
   restricted to Task 7's locked feature set (schema hand-off honoured;
   see the schema-drift test).
3. **Evaluation + metric logging at the end** — `src/pipeline/evaluate.py`,
   validation-only, appended to `outputs/experiments/experiment_log.csv`.
4. **One command** — `run.py` at the project root, a plain CLI script.
5. **Saved artifacts** — `src/pipeline/artifacts.py::save_run_artifacts`
   writes `pipeline.joblib` (model + preprocessing together, one file),
   `metrics.json`, `run_metadata.json` per run, under `outputs/artifacts/<run_id>/`.
6. **Re-run to confirm identical results** — `run.py --verify-reproducibility`
   runs the entire pipeline twice, independently, and diffs the metrics
   programmatically (not eyeballed) — see `outputs/logs/reproducibility_check.json`.

## Each named pitfall gets its own passing, structural test

| Pitfall (from the study guide) | Test | Result |
|---|---|---|
| Preprocessing applied outside the pipeline | `test_pitfall_preprocessing_is_inside_the_pipeline_object` | Asserts the built object is a real `sklearn.pipeline.Pipeline` containing both `impute` and `model` steps, AND source-inspects `run.py` to confirm it never calls `fit_transform` separately — only ever `pipeline.fit(X_train, ...)` |
| Non-reproducible runs | `test_pitfall_runs_are_reproducible` | Actually invokes `python run.py --verify-reproducibility` as a subprocess (not an in-process shortcut) and asserts the two independent runs' metrics are byte-identical with zero differences |
| No saved artifacts | `test_pitfall_artifacts_are_actually_saved` | Saves artifacts, reloads `pipeline.joblib` from disk in a fresh call, and asserts the reloaded pipeline's predictions match the original exactly |

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| One-command, reproducible pipeline outputting model + metrics | `python run.py` — single command, single sklearn Pipeline, `outputs/artifacts/<run_id>/pipeline.joblib` + `metrics.json` from a real run |
| Real-data quality & correctness (realistic, not toy) | Real 569-row WDBC data with Task 7's vetted 31-feature set, not a toy stub dataset |
| Live verification & evidence | `tests/test_pipeline.py` — 6/6 tests pass; reproducibility and one-command tests actually shell out to `run.py` as a subprocess rather than asserting against in-memory function calls |
| Dependency/failure/edge-case handling | Schema-drift test (a locked feature going missing raises clearly, per the guide's own brainstorming question "what breaks if the input schema changes slightly?"); unknown model name rejected before any fitting is attempted |

## How to run

```bash
pip install -r requirements.txt
python tests/test_pipeline.py              # everything, incl. pitfall + edge-case tests
# or the pipeline directly:
python run.py                               # one run -> outputs/artifacts/default_run/
python run.py --run-id experiment_2         # name the artifact folder
python run.py --verify-reproducibility      # Step 6: run twice, confirm identical metrics
```

## Results from this run (seed=42)

```json
{
  "pr_auc": 1.0, "roc_auc": 1.0, "precision": 1.0,
  "recall": 0.9811, "f1": 0.9905, "accuracy": 0.9882
}
```

**Reproducibility check:** two independent full pipeline runs (`run_a`,
`run_b`) produced these exact same six numbers, `differences: {}` —
confirmed in `outputs/logs/reproducibility_check.json`, answering the
guide's brainstorming question "If you hand this to a teammate, can they
reproduce your numbers?" with evidence rather than an assumption.

Full artifacts: `outputs/artifacts/run_a/` and `outputs/artifacts/run_b/`
(each with `pipeline.joblib`, `metrics.json`, `run_metadata.json`).

## A note on the one weak link (per the guide's own question)

"Which stage is the weakest link?" — honestly, it's Step 2: the pipeline
loads Task 2's already-clean data and Task 7's already-vetted feature
list, so it inherits their leakage checks and feature validation rather
than re-verifying them. If the upstream data or feature-list files were
tampered with independently of the pipeline that produced them, this
stage wouldn't catch it — it trusts the hand-off. The schema-drift test
covers the "column went missing" failure mode, not a "column silently
changed meaning" one.

## External resources needed

**None.** Same offline WDBC data as Tasks 1–7. Only `pip install -r
requirements.txt` needs network access, once.

## Folder structure

```
task8_project/
├── README.md
├── requirements.txt
├── run.py                              # THE one command
├── configs/
│   ├── __init__.py
│   ├── loader.py                       # YAML -> typed Config, sets global seed
│   └── config.yaml
├── data/
│   ├── clean_from_task2.csv            # carried over from Task 2
│   └── locked_feature_set.json         # carried over from Task 7
├── src/
│   ├── __init__.py
│   ├── data/dataset.py                  # Step 2: load + honour Task 7's locked features + split
│   └── pipeline/
│       ├── build.py                     # Step 1: THE single sklearn Pipeline
│       ├── evaluate.py                  # Step 3: metrics + experiment log
│       └── artifacts.py                 # Step 5: save/reload run artifacts
├── tests/
│   └── test_pipeline.py                # live subprocess runs + one test per named pitfall + edge cases
└── outputs/
    ├── artifacts/
    │   ├── run_a/  (pipeline.joblib, metrics.json, run_metadata.json)
    │   └── run_b/  (pipeline.joblib, metrics.json, run_metadata.json)
    ├── experiments/
    │   └── experiment_log.csv
    └── logs/
        └── reproducibility_check.json
```
