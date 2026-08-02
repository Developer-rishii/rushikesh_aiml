# Task 18 — SSO, SCIM & Enterprise Identity (AI/ML scope)
PlaceMux · Sprint D - Enterprise Readiness

Support enterprise identity in ML personalization: recruiter-level and
org-level context without cross-contamination. **The bar: a recruiter's
personalization follows their role and org correctly, and leaves with them.**

## Quick start

```bash
pip install numpy pandas scikit-learn   # already present in this environment
python3 data/generate_logs.py           # 1. generate realistic logs
python3 src/evaluate.py                 # 2. offline eval: scoped vs baseline
python3 tests/run_tests.py              # 3. isolation tests (7/7)
python3 src/demo.py                     # 4. end-to-end demo + failure injection
```

## Folder structure

```
task18/
├── README.md                    # this file
├── DESIGN_DECISIONS.md          # every choice + what was rejected, and why
├── DEFINITION_OF_DONE.md        # checklist mapped to study guide §10 + rubric §11
├── EXPERIMENT_LOG.md            # reproducible run log with real numbers
├── RISKS.md                     # gaps disclosed on purpose, not hidden
├── NEXT_STEPS.md                # study guide §14 "go deeper" follow-ups
├── data/
│   ├── generate_logs.py         # synthetic-but-realistic log generator
│   └── interaction_logs.json    # generated output (10,854 events)
├── src/
│   ├── feature_store.py         # CORE: org/recruiter-scoped signal store
│   ├── identity_lifecycle.py    # CORE: join / move / leave propagation
│   ├── ranking_model.py         # learning-to-rank features + metrics
│   ├── evaluate.py              # scoped vs baseline offline evaluation
│   └── demo.py                  # Stage E: end-to-end + induced failures
├── tests/
│   ├── test_isolation.py        # CORE: 7 isolation tests (Stage D)
│   └── run_tests.py             # dependency-free runner (no pytest offline)
└── artifacts/
    ├── eval_results.json                     # final scoped-vs-baseline numbers
    ├── eval_results_v1_grouped_REJECTED.json # kept as evidence of iteration
    └── demo_transcript.txt                   # live demo output
```

## Headline evidence (reproduce with the commands above, seed=42)

| Metric | Scoped model | Baseline (global, unscoped) | Lift |
|---|---|---|---|
| nDCG@10 | 0.3663 | 0.2822 | **+29.8%** relative |
| MAP | 0.4713 | 0.4064 | **+16.0%** relative |
| precision@10 | 0.4842 | 0.3921 | **+23.5%** relative |

Isolation tests: **7/7 passing** (`tests/test_isolation.py`), including a
recruiter_id-collision-across-orgs case and an intentional-break sanity check.

Demo: a test recruiter is moved org_A → org_B; org_A's personal signal is
verified purged, a stale replayed event tagged with the old org is verified
rejected, and a never-seen recruiter is verified to gracefully fall back to
the org-level aggregate instead of erroring. Full transcript in
`artifacts/demo_transcript.txt`.

## What's honestly NOT done
See `RISKS.md`. Most importantly: **no online A/B validation** (no
production traffic available) and **no fairness audit** — both disclosed
explicitly rather than glossed over, per the study guide's own pitfall list
and the rubric's zero-credit-for-unevidenced-claims rule.
