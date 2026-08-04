# Model Card — ranker_v1

**Tenant:** AcmeFinServ_Pilot
**Created:** 2026-08-04
**Artifact:** models/ranker_v1.joblib

## Training data
- Rows trained on: 22463
- Held-out rows: 7537 across jobs ['J004', 'J006', 'J012', 'J015', 'J016', 'J019', 'J026', 'J027', 'J037', 'J039']
- Features: skill_overlap, experience_years, n_candidate_skills, n_required_skills
- Target: clicked + 3*shortlisted + 10*hired
- Random seed: 42 (fully reproducible)

## Approach & rejected alternatives
GradientBoostingRegressor pointwise proxy (see module docstring for rejected alternatives)

## Offline evaluation vs baseline (skill_overlap)
- nDCG@10 delta: 0.0139
- MAP@10 delta: -0.0047
- Precision@10 delta: -0.11
- Online proxy (hire-capture@10) delta: 0.3

## Fairness (protected_group A vs B)
- Demographic parity ratio (min/max): 0.8993
- Passes 4/5ths rule: True
- See experiments/fairness_report.json for equal-opportunity gap finding.

## Known limitations (see docs/remediation_list.md for full list)
- Pointwise proxy, not listwise LambdaMART (LightGBM unavailable in build env)
- protected_group excluded from features by design, used only for audit
- Trained on synthetic-but-realistic tenant data, not a live enterprise export

## Serving fallback
If this artifact is missing or fails to load, the serving layer falls back
to skill_overlap baseline ranking (verified in experiments/latency_report.json
chaos test) rather than failing the request.
