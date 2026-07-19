# Worked Example — one real input/output pair

## Input (sent to `POST /rank`)
```json
{
  "job": {"job_seniority": 2, "job_urgency_score": 0.6, "job_num_applicants_so_far": 40},
  "candidates": [
    {"candidate_id": "c0", "exp_years": 5, "skill_match": 0.72, "education_score": 0.55,
     "location_match": 1, "past_response_rate": 0.35},
    {"candidate_id": "c1", "exp_years": 1, "skill_match": 0.30, "education_score": 0.40,
     "location_match": 0, "past_response_rate": 0.10}
  ]
}
```

## Output
```json
{
  "ranked_candidate_ids": ["c0", "c1"],
  "scores": [1.42, 0.11],
  "used_fallback": false,
  "fallback_reason": null,
  "server_latency_ms": 27.6
}
```

## Plain-English reason
`c0` ranks above `c1` because the model weighs skill-match, experience,
location match and past responsiveness together, and `c0` is materially
stronger on every one of them (0.72 vs 0.30 skill match, on-location vs
not, 5 vs 1 years' experience, higher historical response rate). None of
these is a single deciding factor on its own — `c0` doesn't win purely on
experience or purely on skill match — it wins because it's stronger across
the whole feature set the model was trained on.

## What happens when the model is unavailable
Same request, but with the model forced down (`POST /admin/model_down`,
exercised live in `results/failure_demo_log.json`):

```json
{
  "ranked_candidate_ids": ["c0", "c1"],
  "scores": [0.609, 0.24],
  "used_fallback": true,
  "fallback_reason": "model_forced_down",
  "server_latency_ms": 2.4
}
```

Same relative order in this example (the heuristic — skill_match and
past_response_rate only — agrees with the model here), but note:
- `used_fallback: true` and a `fallback_reason` are always present so any
  downstream consumer (or an on-call engineer looking at logs) can tell
  a request was served by the cheap path, not silently trust a
  degraded score as if it were the full model's judgment.
- Latency dropped from ~28ms to ~2.4ms — the fallback is deliberately far
  cheaper than the model, which is what keeps the service responsive during
  a real outage (see `results/failure_demo_log.json` for the full
  request-by-request trace of this actually happening, including recovery).
