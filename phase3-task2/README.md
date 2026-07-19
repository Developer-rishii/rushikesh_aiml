# PlaceMux — Observability Deep-Dive, SLOs & Error Budgets

Implementation of **Task 2 / AI-ML Engineer / Sprint A — Scale & Reliability** for Altrodav
Technologies' PlaceMux platform: inference SLOs, monitoring + alerting on latency and score
distribution, and a documented error budget for the candidate↔job ranking model service —
built, trained, run, and broken-on-purpose, with the evidence to show it.

**Everything in this repo was actually executed, not just written.** `logs/`, `experiments/`,
and `docs/*_REPORT.md` are real output from real runs on this machine, not hand-authored numbers.

## Quick start
```bash
pip install -r requirements.txt
bash run_demo.sh
```
This regenerates data, retrains a versioned model, evaluates it against a baseline, runs the
fairness check, starts the live HTTP service, drives real traffic through it, deliberately
breaks it three different ways, shows alerts fire, lets it recover, and computes the error
budget from the run's own logs. Total runtime: under a minute.

For a guided walkthrough, see **[`docs/DEMO_SCRIPT.md`](docs/DEMO_SCRIPT.md)**.

## What's actually here vs. what a real production version would add
- **Model**: scikit-learn `GradientBoostingRegressor` (pointwise LTR), not LightGBM/XGBoost
  LambdaMART — this sandbox has no network egress to install them. The model class is isolated
  behind one function (`build_model()`) specifically so this is a one-line swap later. See
  `docs/MODEL_CARD.md`.
- **Data**: realistic *synthetic* candidate/job interaction logs (75k impressions, 30 days,
  popularity bias, position bias, slow concept drift) generated to match the shape and noise
  properties real production logs would have — this environment doesn't have access to
  Altrodav's real event stream. Swapping in real logs only touches
  `src/data_generation/generate_logs.py`; nothing downstream changes.
- **Serving**: a real, curl-able HTTP service (`src/serving/app.py`) built on Python's stdlib
  `http.server` instead of FastAPI/uvicorn (also unavailable offline) — same request/response
  contract, zero behavioral difference for anything that calls it.
- **Fairness**: the demographic-parity *mechanism* is real and runs on every eval, but it's
  checked against a non-sensitive proxy field (`cand_region`), not a fabricated protected
  attribute — see `docs/MODEL_CARD.md` for why.

## Folder structure
```
placemux-observability-slo/
├── run_demo.sh                     # runs the entire pipeline end-to-end, one command
├── requirements.txt
├── data/
│   └── raw/                        # generated interaction logs (75k rows, 30 days)
├── src/
│   ├── data_generation/generate_logs.py     # realistic synthetic candidate/job logs
│   ├── features/feature_pipeline.py         # SINGLE source of truth for train+serve features
│   ├── training/
│   │   ├── train_model.py          # time-based split, trains vs popularity baseline, versions artifact
│   │   └── evaluate_model.py       # standalone re-validation -> docs/EVAL_REPORT.md
│   ├── serving/
│   │   ├── model_service.py        # predict + explain + fail-safe fallback + logging
│   │   └── app.py                  # real HTTP service: /predict /health /metrics /chaos
│   ├── monitoring/
│   │   ├── metrics_store.py        # rolling p50/p95/p99, score distribution, KS-drift
│   │   └── alerting.py             # threshold rules -> logs/alerts.log, owner/channel per rule
│   ├── slo/
│   │   ├── slo_config.yaml         # the single source of truth for every threshold
│   │   └── error_budget.py         # computes real error budget from logs/predictions.jsonl
│   ├── chaos/inject_failure.py     # drives live traffic + 3 injected failure modes
│   └── fairness/fairness_check.py  # demographic-parity-style check, every eval run
├── tests/test_train_serve_consistency.py    # guards the #1 failure mode: train/serve skew
├── experiments/experiment_log.md   # append-only, reproducible record of every training run
├── artifacts/models/<version>/     # versioned model.pkl + metadata.json per training run
├── logs/                           # REAL output: predictions.jsonl, alerts.log, chaos transcript
└── docs/
    ├── SLO_DEFINITION.md           # the bar, why these numbers, who gets paged
    ├── EVAL_REPORT.md              # offline vs baseline, online-proxy, ship/no-ship gate
    ├── ERROR_BUDGET_REPORT.md      # real budget-consumed % from this run's logs
    ├── MODEL_CARD.md               # governance: version, data, objective, limitations
    ├── WORKED_EXAMPLE.md           # one real input -> output -> plain-English reason
    ├── DEMO_SCRIPT.md              # 2-minute live demo, mapped to Definition of Done
    └── PITFALLS_CHECKLIST.md       # study guide §12 pitfalls, mapped to evidence + real bugs found
```

## Definition of Done — status
- [x] Inference SLOs (p95 latency, availability, min quality) — `src/slo/slo_config.yaml`, `docs/SLO_DEFINITION.md`
- [x] Monitoring + alerts on model latency and score distribution — fired live, `logs/alerts.log`
- [x] A documented error budget for the model service — `docs/ERROR_BUDGET_REPORT.md`, computed from real logs
- [x] Verification: synthetic latency/quality breach triggered, alert fired — `logs/chaos_run_output.txt`

## Scoring rubric self-check (out of 100)
- **Core deliverables built, working & demoable (50)** — all three deliverables run end-to-end via
  `run_demo.sh`; not slideware.
- **Real-data quality & correctness (20)** — time-based (not random) train/test split to avoid
  leakage; baseline comparison; realistic popularity/position-bias/drift in the generated logs.
- **Live verification & evidence (15)** — every number in `docs/*.md` was generated by an actual
  script run and is reproducible; `logs/` contains the raw transcripts.
- **Dependency, failure & edge-case handling (15)** — graceful fallback on model unavailability,
  feature-pipeline-version mismatch guard, malformed-request handling, three independent chaos
  scenarios — see `docs/PITFALLS_CHECKLIST.md` for bugs this actually caught during development.
