# Worked Example (Explainability)

**This input:** job `job_00001`, 40 shortlisted candidates in the held-out test set.

**This output:** the optimized (AFTER) server ranked candidate `job_00001_c036`
#1, with predicted score `1.322` (true logged relevance for this candidate was
`1`, i.e. clicked).

**This plain-English reason:** Ranked #1 mainly due to `skill_overlap`
(value = 0.92, global feature importance = 0.70 across the whole model),
combined with a strong `skill_overlap` and 13.6 years of `years_experience`.

**Full feature-importance breakdown of the optimized model** (from
`RandomForestRegressor.feature_importances_`, i.e. read directly off the
trained model, not asserted):

| Feature | Importance |
|---|---|
| skill_overlap | 0.703 |
| candidate_activity_score | 0.085 |
| years_experience | 0.068 |
| distance_km | 0.062 |
| candidate_past_response_rate | 0.027 |
| job_fill_urgency | 0.023 |
| salary_gap_pct | 0.020 |
| recruiter_rating | 0.011 |

This confirms the model is dominated by the feature the data-generating process
actually made most predictive (`skill_overlap`), which is what we'd want to see
in a sanity check before trusting the ranking in production.

## What happens when the model is unavailable

See `experiments/metrics_before_after.json -> failure_injection_demo` and the
"Failure injection" section of `before_after_report.md`: the request is still
served, via a cheap non-ML popularity heuristic, instead of erroring out.
