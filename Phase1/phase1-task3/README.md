# Task 3 — Core Architecture

PlaceMux · Altrodav Technologies · AI/ML Developer · Phase 1

**Continuation of Task 1 & 2:** this project loads Task 2's leakage-cleaned,
balance-checked dataset (`data/clean_from_task2.csv`, copied from
`phase1-task2/data/clean.csv`) and reuses `SEED=42` and `target_col="target"`
throughout, so all three tasks stay one coherent pipeline.

## What this delivers (maps to the brief's Definition of Done)

A **modular, config-driven train/eval skeleton with the baseline running
through it**, demonstrable live on real data:

- `src/data/` — load + reproducible stratified split
- `src/features/` — config-driven preprocessing (impute/scale/drop)
- `src/models/` — a **registry**: one place to add a new model
- `src/evaluate/` — one place metrics are computed, one place runs get logged
- `src/harness.py` — the single train/evaluate harness every model plugs into
- `configs/config.yaml` — controls every path, param, and seed; nothing
  in `src/` hardcodes any of these

## Proof this is real, not just structured to look real

Two full runs were executed for this submission, differing **only** in
which config file was passed — no source file touched between them:

```bash
python -m src.harness --config configs/config.yaml               # logreg_baseline
python -m src.harness --config configs/config_random_forest.yaml  # random_forest
```

Both rows landed in `outputs/experiments/experiment_log.csv`, both model
artifacts were saved to `outputs/models/`. That's the "swap models without
rewrites" requirement demonstrated, not asserted.

## How this maps to the scoring rubric (100 pts)

| Rubric item | Where it's satisfied |
|---|---|
| Reproducible env, correct split, working smoke-test run w/ logged metrics | `SEED=42` set once in `configs/loader.py::set_global_seed`; stratified 70/15/15 split with an index-overlap leakage guard in `src/data/dataset.py`; `outputs/experiments/experiment_log.csv` + `outputs/logs/run_*.log` from real runs |
| Real-data quality & correctness (realistic, not toy) | Real 569-row WDBC clinical data, already leakage-cleaned in Task 2, carried through unchanged |
| Live verification & evidence | `tests/test_harness.py` — 6/6 tests pass on real runs, including a real model-swap-via-config test; log files and `.joblib` artifacts are the actual run output |
| Dependency/failure/edge-case handling | Every harness stage (data/features/model/evaluate) wrapped with a specific error message and `sys.exit(1)`; tests cover unknown model name, missing config, and invalid split fractions |

## How to run

```bash
pip install -r requirements.txt
python tests/test_harness.py                                   # everything, incl. edge cases
# or individually:
python -m src.harness                                           # baseline via default config
python -m src.harness --config configs/config_random_forest.yaml  # swap model, same code
```

## How to add a new model (documented per the brief's Step 6)

1. In `src/models/registry.py`, write a factory function:
   ```python
   def build_my_model(params: dict):
       return MyEstimator(**params)   # any sklearn-compatible estimator
   ```
2. Add it to `REGISTRY = {..., "my_model": build_my_model}`.
3. In a config YAML, set `model.name: "my_model"` and `model.params: {...}`.
4. Run `python -m src.harness --config <your_config>.yaml`.

No other file changes. `src/data/`, `src/features/`, `src/evaluate/`, and
`src/harness.py` are all model-agnostic by construction.

## Pitfalls avoided (per the brief's §Pitfalls)

| Pitfall | How it's avoided |
|---|---|
| Hard-coded paths/params | Everything routes through `configs/config.yaml` -> `configs/loader.py::Config`; grep the codebase, there's no literal path/seed outside those two files |
| Copy-paste experiments | One harness (`src/harness.py`) for every model; a new experiment is a new YAML file, not a new script |
| Eval logic duplicated/inconsistent | `src/evaluate/metrics.py::compute_metrics` is the only place any metric is computed, called by the harness and by every test |

## Results from this run (seed=42)

- **logreg_baseline:** val PR-AUC 1.0, recall 0.962, accuracy 0.977
- **random_forest:** val PR-AUC 1.0, recall 0.962, accuracy 0.977
  (both strong — WDBC is a well-separated dataset, as already noted in Task 2)

Full numbers: `outputs/experiments/experiment_log.csv`.

## External resources needed

**None.** Same offline WDBC data carried over from Tasks 1–2. Only
`pip install -r requirements.txt` needs network access, once.

## Folder structure

```
task3_project/
├── README.md
├── requirements.txt
├── configs/
│   ├── __init__.py
│   ├── loader.py                    # YAML -> typed Config, sets global seed
│   ├── config.yaml                  # default: logreg_baseline
│   └── config_random_forest.yaml    # same skeleton, different model
├── data/
│   └── clean_from_task2.csv         # carried over from Task 2
├── src/
│   ├── __init__.py
│   ├── harness.py                   # THE single train/eval harness
│   ├── data/
│   │   └── dataset.py                # load + stratified split + leakage guard
│   ├── features/
│   │   └── build.py                  # config-driven preprocessing transformer
│   ├── models/
│   │   └── registry.py               # add-a-model-here registry
│   └── evaluate/
│       ├── metrics.py                # single source of metric computation
│       └── experiment_log.py         # single source of run logging
├── tests/
│   └── test_harness.py               # live e2e run, model-swap proof, edge cases
└── outputs/
    ├── experiments/
    │   └── experiment_log.csv        # generated: append-only run log
    ├── models/
    │   ├── logreg_baseline.joblib
    │   └── random_forest.joblib
    └── logs/
        ├── run_logreg_baseline.log
        └── run_random_forest.log
```
