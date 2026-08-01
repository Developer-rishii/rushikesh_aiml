# PlaceMux Partner API — ML Scoring & Matching

Base URL (local demo): `http://localhost:8000`

## Authentication

Send your key on every request:

```
X-API-Key: <your-key>
```

Demo keys (see `api/auth.py`):
| Key | Partner | Plan | Rate limit |
|---|---|---|---|
| `ats-demo-key-001` | DemoATS Inc. | standard | 10 req/min, 2000/day |
| `ats-gold-key-002` | GoldPartner Corp. | premium | 60 req/min, 20000/day |

## Versioning

Every scoring endpoint is prefixed with a version: `/v1/...`, `/v2/...`.
**A version's model behaviour never changes silently.** New models are published
under a new version. Pin to the version you've integration-tested against; you
control when you move to a newer one.

## `GET /v1/health`
Returns service status and which model versions are currently published.

## `GET /v1/quota`
Returns your remaining requests for the current minute window.

## `POST /{version}/score`
Score a single candidate against a job.

**Request**
```json
{
  "candidate_id": 501,
  "features": {
    "skill_overlap": 5,
    "seniority_gap": 0,
    "same_location": 1,
    "recency_days": 1,
    "candidate_activity": 0.95
  }
}
```

**Response `200`**
```json
{
  "model_version": "v1",
  "model_hash": "efbcc088057629d3",
  "score_band": 80,
  "reasons": ["Strong shared skills with the role", "Strong seniority match with the role"],
  "degraded_mode": false,
  "served_at": "2026-07-31T16:18:29Z"
}
```

- `score_band` — an integer 0–100 in steps of 5. **Not** the raw model output
  (see [Explanation Contract](#explanation-contract) below for why).
- `reasons` — up to 3 plain-English factors, ranked by contribution. We never
  return raw feature weights or your exact feature vector back.
- `degraded_mode` — `true` if the live model was unavailable and a transparent
  rule-based fallback was used instead of failing your request.

## `POST /{version}/match`
Rank many candidates for one job in a single call.

**Request**
```json
{
  "job_id": 77,
  "candidates": [
    {"candidate_id": 1, "features": {"...": "..."}},
    {"candidate_id": 2, "features": {"...": "..."}}
  ]
}
```

**Response `200`** — `ranked_candidates`, an array of the same per-candidate
objects as `/score`, sorted by `score_band` descending.

## Errors

| Status | Body `error` | Meaning |
|---|---|---|
| 401 | `invalid_or_missing_api_key` | Missing/unknown `X-API-Key` |
| 429 | `rate_limit_per_minute_exceeded` | Slow down, retry after ~60s |
| 429 | `daily_quota_exceeded` | Contact us to raise your plan |
| 429 | `abuse_pattern_detected_broad_scraping` | Too many distinct candidates queried too fast — this looks like model extraction, not normal ATS traffic. Contact support if this was legitimate. |

All rejections include a `limits` object so you can back off correctly.

## Explanation contract

**We promise:** a coarse 0–100 score band, up to 3 plain-English reasons, and
the model version/hash that produced the decision.

**We never expose:** raw model weights or feature importances, your exact
input feature vector, or full-precision scores. This is a deliberate
anti-extraction measure — coarse, bucketed outputs give an attacker far less
signal per query, which materially slows down model-cloning attempts, while
still giving you everything you need to explain a match to a candidate.

## What happens if the model is down?

You still get a `200` with `degraded_mode: true` and a rule-based fallback
score/explanation. We degrade service quality before we degrade availability.
We never return a bare `500` for a model outage.
