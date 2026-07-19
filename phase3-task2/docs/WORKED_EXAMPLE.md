# Worked Example — One Real Prediction, End to End

Model version: `model_v20260719_140957` · Feature pipeline: `fp_v1.0.0`

## Input (real row from the held-out interaction log)

```json
{
  "candidate_id": "C00135",
  "job_id": "J0769",
  "cand_experience_yrs": 7.29,
  "cand_expected_salary": 11.97,
  "cand_region": "East",
  "cand_skills": "react|sap|tally",
  "job_min_exp": 3.40,
  "job_salary_offered": 6.76,
  "job_region": "East",
  "job_req_skills": "accounting|java|seo",
  "job_popularity": 5.94
}
```

## Output

```json
{
  "score": 0.5808,
  "explanation": "Score was mainly raised by 'exp_gap' (value=3.895, feature_importance=0.333).",
  "model_version": "model_v20260719_140957",
  "feature_pipeline_version": "fp_v1.0.0",
  "latency_ms": 1.927,
  "degraded": false,
  "success": true
}
```

## Plain-English reason

The candidate has 7.3 years of experience against a job that only needs 3.4 — a comfortable
positive experience gap, which is the single largest factor pushing this score up
(feature importance 0.333, the second-highest weighted feature overall after skill overlap).
Region matches exactly (East/East). Skill overlap with the job's requirements
(accounting/java/seo) is low, which is why this is a moderate rather than a high score — this
candidate is over-qualified on experience but not a strong skills match for this specific role.

## What happens when the model is unavailable

Same input, model forced unavailable (chaos-injected):

```json
{
  "score": 1.9376,
  "explanation": "FALLBACK (popularity baseline) used -- primary model unavailable: chaos: model process forced unavailable",
  "model_version": "model_v20260719_140957",
  "feature_pipeline_version": "fp_v1.0.0",
  "latency_ms": 0.011,
  "degraded": true,
  "success": true
}
```

The service does **not** 5xx or hang — it fails over to the popularity baseline
(`log1p(job_popularity)`), returns instantly, and marks the response `degraded: true` so
monitoring and the error budget both see that the service is "up" but running below its
normal quality bar. This is exactly what `docs/ERROR_BUDGET_REPORT.md`'s "degraded/fallback
responses count as partial burn" policy line is tracking.

## Live verification this actually happened

This exact scenario was run against the live HTTP service (not just this in-process call) as
part of `src/chaos/inject_failure.py` — see `logs/chaos_run_output.txt` for the full transcript,
`logs/predictions.jsonl` for every individual request logged, and `logs/alerts.log` for the
alerts it triggered (`LATENCY_P95_BREACH`, `SCORE_DEGENERATE`, `SCORE_DRIFT`).
