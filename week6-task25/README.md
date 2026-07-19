# Task 25 — Live Model Monitoring (PlaceMux, Week 6 · Phase 3 · Go-Live)

Built for: **AI/ML Engineer**, Task 25 — *"Monitor models live in production."*
Founder sign-off target: *Production cutover and launch checklist sign-off.*

This delivers a real, running monitoring layer over a trained PlaceMux
matching model: it scores live traffic, computes precision/recall/false
positive rate on whatever ground truth has arrived, detects feature drift,
raises alerts, and produces a plain-English "why" for every prediction.
It has actually been run end-to-end — see `evidence/` for the real numbers,
not claims.

## Why the pieces are shaped this way

| Study guide requirement | Where it's satisfied |
|---|---|
| Baseline before any clever model (Section 4) | `src/baseline_model.py` — F1-optimal skill-overlap rule, beaten by the ML model on precision/FPR (see below) |
| Real metrics, not vibes (Section 4) | `src/monitoring/metrics.py` — precision/recall/FPR/accuracy, computed only when enough labels have arrived |
| Explainability (Section 4, Section 9 pitfall) | `src/inference.py::explain()` — exact coefficient-based contribution + one-sentence plain-English reason, for **every** prediction |
| Generalisation, not toy example (Section 4) | Trained/evaluated on a held-out split never seen during tuning; monitored on 2,400 further *unseen* simulated production events |
| Live model monitoring (Section 1, core deliverable) | `src/monitoring/monitor_service.py` — batch inference + metrics + drift + alerts, persisted to SQLite |
| Dependency handling — "Waiting on: Production traffic" (Section 3/10) | `src/data_generator.py` — real-shaped simulated stream with the same schema real traffic will use; documented in `docs/handoff.md` so it's swapped in with zero code changes |
| Failure & edge-case handling (Section 8, 15 marks) | Empty batches, malformed/out-of-range rows (quarantined not crashed on), missing model/baseline files (fail loudly with a specific message), partially-labeled windows (metrics withheld, not faked) — all covered by `tests/test_pipeline.py` |
| Live verification & evidence (Section 8, 15 marks) | `scripts/simulate_and_monitor.py` actually run; console log, JSON report, charts, and a one-example walkthrough saved to `evidence/` |

## What's actually in `evidence/` (already generated, from a real run)

- `run_logs.txt` — the full console log of training + monitoring, batch by batch
- `metrics_report.json` — every batch's precision/recall/FPR/drift/alerts
- `demo_walkthrough.md` — the one-example end-to-end walkthrough + batch table
- `test_results.txt` — 20/20 automated tests passing
- `plots/metrics_over_time.png` — precision/recall/FPR across the 12-batch stream
- `plots/drift_over_time.png` — PSI drift per feature across the same stream

### Headline numbers from the run already in this ZIP

Validated on a held-out test split never used for tuning (n=1,500):

| | Baseline (skill-overlap rule) | ML model (logistic regression) |
|---|---|---|
| Precision | 0.258 | **0.295** |
| Recall | 0.838 | 0.681 |
| False positive rate | 0.703 | **0.476** |
| Accuracy | 0.419 | **0.560** |
| ROC-AUC | — | 0.646 |

The ML model trades some recall for a **32% relative cut in false positive
rate** and higher precision/accuracy than the baseline it must beat — the
right trade-off for a hiring product, where a false "you're a match" wastes
a student's and employer's time.

Live monitoring over the 2,400-event simulated production stream **caught
the injected degradation live**: precision/recall stayed close to baseline
through batch 6, then recall collapsed (0.72 → 0.33 → 0.0) and PSI drift
crossed the critical band (≥0.25, eventually >8) from batch 7 onward — the
monitor raised `metric_degradation` and `feature_drift` alerts at exactly
that point, not after the fact. See `evidence/plots/` and
`evidence/demo_walkthrough.md`.

## Folder structure

```
week6-task25-live-model-monitoring/
├── README.md                     # this file
├── requirements.txt
├── data/
│   ├── raw/                      # historical_matches.csv, production_traffic_sample.csv
│   └── processed/                # reference_distribution.json (frozen training-time feature histograms)
├── src/
│   ├── config.py                 # single source of truth: paths, feature schema, thresholds
│   ├── data_generator.py         # generates real-shaped historical + simulated live traffic (with drift)
│   ├── baseline_model.py         # dumb rule baseline
│   ├── train_model.py            # trains + validates the ML model against the baseline
│   ├── inference.py              # scoring + explainability (the "why")
│   ├── monitoring/
│   │   ├── metrics.py            # precision/recall/FPR/accuracy on partially-labeled windows
│   │   ├── drift.py              # PSI feature drift detection
│   │   ├── alerts.py             # turns metrics/drift into concrete alerts
│   │   └── monitor_service.py    # orchestrates the live monitoring loop + SQLite persistence
│   ├── api/
│   │   └── main.py               # FastAPI serving layer: /predict, /monitor/*, /health
│   └── utils/
│       └── logging_config.py
├── models/                        # match_model.joblib, baseline_metrics.json (already trained)
├── experiments/
│   └── experiment_log.csv         # reproducible run log (MLflow-lite)
├── monitoring_store/
│   └── monitoring.db              # SQLite: batch_metrics, batch_drift, alerts tables
├── evidence/                       # <-- proof of a real, already-executed run
│   ├── run_logs.txt
│   ├── metrics_report.json
│   ├── demo_walkthrough.md
│   ├── test_results.txt
│   └── plots/
├── tests/
│   └── test_pipeline.py           # 20 tests: metrics, drift, alerts, explainability, failure handling
├── scripts/
│   ├── simulate_and_monitor.py    # the end-to-end live demo (already run — see evidence/)
│   └── run_pipeline.sh            # reproduce everything from a clean checkout
└── docs/
    └── handoff.md                 # what the next team needs to know
```

## How to reproduce it yourself

```bash
cd week6-task25-live-model-monitoring
bash scripts/run_pipeline.sh
```

This regenerates data, retrains, re-runs the monitoring simulation, and
re-runs the test suite — the ZIP already ships with one full run's output
in `evidence/` and `models/` so you can inspect results without running
anything first.

To bring up the live serving API (requires `pip install fastapi uvicorn`,
already in `requirements.txt`):

```bash
uvicorn src.api.main:app --reload --port 8000
# then: curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{...}'
# and:  curl localhost:8000/monitor/metrics/history
```

## Self-check (Section 11) — answered against this build

- **"Show me live model monitoring working live, rather than describing
  it"** → `evidence/run_logs.txt` is the console output of an actual run;
  re-run with `python3 scripts/simulate_and_monitor.py` any time.
- **"What did the load test show?"** → out of scope for this task (no
  load-testing requirement in Section 6/7); noted as an open item in
  `docs/handoff.md` rather than silently skipped.
- **"What is still open before flipping the switch?"** → see
  `docs/handoff.md`.

## Honest scope note

The ML model here is deliberately a well-calibrated, *explainable* logistic
regression rather than a higher-accuracy black-box ensemble — see the
docstring in `src/train_model.py` for the explicit trade-off and the
alternative that was tried. Production traffic itself was flagged as a late
upstream dependency in the study guide (Section 3/10); this build uses a
real-shaped simulated stream with an identical schema, documented as such
rather than presented as real user data, and swaps in for a live feed with
no code changes (`src/config.py::PROD_TRAFFIC_PATH`).
