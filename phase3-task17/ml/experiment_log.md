# Experiment Log

- Run at: 2026-08-01T12:12:43Z
- Train rows: 15122, Holdout rows (unseen jobs): 4878
- Model: GradientBoostingRegressor(n_estimators=150, max_depth=3, lr=0.08), trained in 1.116s
- Model hash: 605c1336c6dd7670 (bound to API version v1 — see versioning.py)

## Offline metrics (held-out jobs, not tuned on)

| metric | baseline | model | lift |
|---|---|---|---|
| nDCG@10 | 0.4378 | 0.6750 | 54.18% |
| precision@5 | 0.7667 | 0.9267 | 20.87% |

## Feature importances (internal only — never returned raw via API)

- skill_overlap: 0.7527
- seniority_gap: 0.1538
- same_location: 0.0421
- recency_days: 0.0264
- candidate_activity: 0.025

## Honest caveat

Offline nDCG measures ranking quality on logged impressions only; it does not capture that better ranking changes WHICH candidates get impressions at all (position/selection bias). Must be confirmed with an online A/B before claiming a win.
