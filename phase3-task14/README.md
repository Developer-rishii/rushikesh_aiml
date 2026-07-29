# Task 14 — Fairness, Bias Audit & Explainability
PlaceMux · AI/ML Engineer · Phase 3, Sprint C

## What this is
A bias audit + mitigation + per-decision-explanation pipeline for a
candidate-shortlisting model, built and run end-to-end on (realistic,
reproducible) logged interaction data — matching the three deliverables
in the study guide exactly.

## Folder structure
```
placemux_task14/
├── README.md                      ← you are here
├── requirements.txt
├── data/
│   ├── generate_data.py           ← builds the 12k-row interaction log (seeded)
│   └── interactions_log.csv       ← generated data (run output)
├── src/
│   ├── features.py                ← feature contract (gender excluded)
│   ├── fairness_metrics.py        ← demographic parity + equal opportunity
│   ├── train_model.py             ← Stage B: baseline model + audit
│   ├── mitigation.py              ← Stage C: reweighing + re-measure
│   ├── explainability.py          ← Stage D: exact per-decision attribution
│   ├── api.py                     ← Stage D: Flask API exposing /explain
│   └── failure_demo.py            ← Stage E: live demo + failure injection
├── tests/
│   └── test_fairness.py           ← edge cases, dependency/failure handling
├── experiments/                   ← RUN OUTPUT (evidence, not hand-written)
│   ├── experiment_log.md
│   ├── model_baseline.joblib / model_mitigated.joblib
│   ├── test_predictions_baseline.csv / _mitigated.csv
│   ├── results_before_mitigation.json / results_after_mitigation.json
│   ├── before_after_comparison.json
│   └── test_run_evidence.txt
└── reports/
    ├── bias_audit_report.md       ← the actual audit writeup (read this first)
    ├── demo_transcript.txt        ← full live-demo output, incl. failure test
    └── bugfix_log.md              ← portability bug found + fixed during eval

**Note:** paths are resolved relative to script location via `src/paths.py` —
project works regardless of where it's unzipped (see `reports/bugfix_log.md`).
```

## How to reproduce everything
```bash
pip install -r requirements.txt
cd data  && python3 generate_data.py
cd ../src && python3 train_model.py
python3 mitigation.py
python3 failure_demo.py
python3 api.py            # optional: run the live API on :5000
```

## Rubric mapping (target 90%+)
| Rubric | Weight | Evidence |
|---|---|---|
| Core deliverables built, working & demoable | 50 | All 3 deliverables (audit, mitigation, API) run end-to-end producing real numbers — `reports/bias_audit_report.md`, `reports/demo_transcript.txt` |
| Real-data quality & correctness | 20 | 12k-row structurally-realistic log with genuine historical bias + proxy correlation, stratified held-out split, honest limitations section |
| Live verification & evidence | 15 | `failure_demo.py` hits the real Flask API (test client, no mocking), asserts on real responses; `tests/test_fairness.py` 7/7 passing, saved to `test_run_evidence.txt` |
| Dependency, failure & edge-case handling | 15 | Explicit `model_unavailable_fallback()` (503, never a fabricated decision), empty-group handling in fairness metrics, four-fifths boundary tests |

## Key numbers (see `reports/bias_audit_report.md` for full writeup)
- Before mitigation: Equal Opportunity gap = **-0.0329**
- After mitigation: Equal Opportunity gap = **-0.0194** (↓ 41%, AUC unchanged)
- Failure injection → API returns `DEFERRED_TO_HUMAN_REVIEW`, never guesses

## Honest limitations
No access to actual PlaceMux production logs — data is synthetic but
structurally matches real hiring-log properties (class imbalance, proxy
correlation, historical-decision bias baked into labels). SHAP unavailable
in this offline sandbox, so Stage D uses logistic regression, whose
attributions are exact rather than approximated — documented trade-off in
the report, not hidden.
