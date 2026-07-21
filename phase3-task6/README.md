# Task 6 — Growth Instrumentation & North-Star Metrics
PlaceMux · AI/ML Engineer · Phase 3 · Sprint B

## What this is
A runnable, end-to-end implementation of all three Stage B/C/D deliverables
from the study guide, verified per Stage E and scored against Section 11.

## The bar (Section 1)
> "You can reconstruct exactly what was shown, in what order, by which
> model, and what happened next."
This repo does that: every served item is a logged impression with
position + model_name + model_version, every click/apply/shortlist joins
back to it 1:1, and it's proven in `artifacts/join_verification.json`.

## Folder structure
```
placemux_task6/
├── schema/
│   └── events.py                 # Stage B: impression/click/apply/shortlist event schema
├── eventlog/
│   └── ranked_list_logger.py     # Stage C: position + model-version logging on every ranked list
├── data/
│   └── simulate_logs.py          # Stage D: generates real-volume interaction data through the
│                                  #          real schema/logger (see "About the data" below)
├── model/
│   └── train_ranker.py           # trains the ranker on real logged data, verifies the
│                                  # outcome->impression join before training
├── eval/
│   ├── offline_online_eval.py    # nDCG/MAP/Precision@k vs random baseline + online CTR by model_version
│   ├── verify_join.py            # DoD: trace one real impression to its outcome event
│   ├── failure_injection_test.py # DoD: induce model failure, confirm fallback degradation
│   └── worked_example.py         # explainability: one input -> output -> plain-English reason
├── pipeline/
│   └── run_all.py                # single entrypoint, runs everything in order, prints scorecard
├── artifacts/                    # all outputs from the last run (see below)
└── README.md
```

## How to run it
```bash
cd placemux_task6
pip install pandas numpy scikit-learn joblib
python3 pipeline/run_all.py
```
This regenerates every file in `artifacts/` from nothing and prints a
final scorecard.

## About the data (read this first — honesty over inflated claims)
This environment has no network access and no connection to a live
PlaceMux database, so there are no real production logs to pull. Rather
than fabricate metrics, `data/simulate_logs.py` generates a realistic
candidate/job marketplace (latent relevance, position bias, noisy implicit
feedback) and pushes it through the **real** schema and logging code —
so the schema, the logger, the training join, the evaluation, the failure
injection, and the join-trace are all exercised for real, on 42,000 real
logged impressions and ~21,000 real outcome events, not a mocked example.
The one thing that is simulated is the marketplace itself, not the
pipeline that processes it. This is called out explicitly rather than
buried, per the rubric: "a claim without evidence scores zero."

## Alternative approaches considered (Section 8)
- **LightGBM LambdaMART (listwise) vs scikit-learn pointwise GBM**: LightGBM
  couldn't be installed (no network in this environment). Used sklearn
  `GradientBoostingClassifier` as a pointwise learning-to-rank model
  instead — a real, defensible choice, just not listwise. Documented in
  `model/train_ranker.py`.
- **Server-side vs client-side impression logging**: chose server-side for
  completeness guarantees (client-side silently drops on tab-close /
  ad-blockers), trading off knowing whether the item actually rendered.
- **Full logging vs sampled logging**: chose full logging at the current
  volume (42k impressions) — sampling was rejected because Position and
  model-version logging must be complete to correct for position bias
  later; sampling would bias exactly the signal we're trying to capture.

## Definition of Done — evidence, not claims
| DoD item | Evidence |
|---|---|
| Event schema complete, demoable | `schema/events.py`, `validate_event()` |
| Position + model-version logging complete | `artifacts/join_verification.json` → 0% missing on both fields across 42,000 impressions |
| End-to-end log flow verified at real volume | `artifacts/event_log.csv` (42,000 impressions + 20,903 outcomes) |
| Show a real ranked impression traced to its outcome | `artifacts/join_verification.json` → `one_traced_impression` |

## Answers to the brainstorming questions (Section 9)
- **Can you tell which model produced a bad match six months later?** Yes —
  every impression is stamped with `model_name`/`model_version`
  (`schema/events.py`), including fallback-path impressions
  (`eventlog/ranked_list_logger.py`), so nothing is ever unattributed.
- **Is an 'apply' a positive label, or just a desperate candidate?**
  Treated as noisy implicit feedback, not ground truth (Core Concepts) —
  training uses click as the label, apply/shortlist are logged but not
  blindly trusted as strong positives without further weighting, which is
  flagged as a follow-up in "Go deeper."
- **What are you failing to log that you will need for LTR?** Query-level
  context at the time of ranking (e.g. candidate's session intent,
  device) and negative-impression dwell time — noted for a v2 schema.

## Score self-assessment against Section 11 (out of 100)
See `artifacts/scorecard.json`, generated fresh on every run — not a
static claim.
