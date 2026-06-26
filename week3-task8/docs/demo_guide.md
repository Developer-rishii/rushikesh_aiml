# Spend-Quality Guardrail — Live Demo Guide

This guide walks you through testing the guardrail API live. All examples below use real data and real model output from the current build.

## Current Calibrated Performance

*From `metrics/guardrail_metrics.json` — evaluated on held-out 15% test set (300 records):*

| Metric              | Dumb Baseline (warn everyone) | Calibrated Guardrail |
|---------------------|-------------------------------|----------------------|
| Precision           | 0.7200                        | **0.8966**           |
| Recall              | 1.0000                        | 0.8426               |
| Accuracy            | 0.7200                        | **0.8167**           |
| F1 Score            | 0.8372                        | **0.8687**           |
| False Positive Rate | 1.0000                        | **0.2500**           |

**Threshold used**: 55% (prediction_score from LogisticRegression)

**Why this threshold**: F1 maximization balances blocking genuinely bad matches (recall) against not wrongly blocking good matches (precision). At 55%, precision improves by +17.7pp over the dumb baseline while retaining 84.3% recall.

---

## 1. Good Match Walkthrough (Passes Guardrail)

**Request**:
```bash
curl -X POST http://127.0.0.1:8000/guardrail-check \
  -H "Content-Type: application/json" \
  -d '{"candidate_id": 1, "job_id": 19}'
```

**Actual Response** (status 200):
```json
{
  "candidate_id": 1,
  "job_id": 19,
  "match_score": 72.03,
  "fit_status": "OK",
  "threshold_used": 55,
  "reason": [
    "Match score 72.0% passes the 55.0% spend-safety threshold.",
    "Matched skills: C++, Git, SQL",
    "Missing required skills: Machine Learning, React",
    "Education level does not match the job requirement."
  ]
}
```

**Outcome**: `fit_status` is `OK`. The checkout flow should authorize the application spend.

---

## 2. Bad Match Walkthrough (Fails Guardrail)

**Request**:
```bash
curl -X POST http://127.0.0.1:8000/guardrail-check \
  -H "Content-Type: application/json" \
  -d '{"candidate_id": 1, "job_id": 1}'
```

**Actual Response** (status 200):
```json
{
  "candidate_id": 1,
  "job_id": 1,
  "match_score": 16.88,
  "fit_status": "LOW_FIT_WARNING",
  "threshold_used": 55,
  "reason": [
    "Match score 16.9% is below the 55.0% spend-safety threshold.",
    "Matched skills: C++",
    "Missing required skills: AWS, Docker, Machine Learning",
    "No matching certifications with the job requirements."
  ]
}
```

**Outcome**: `fit_status` is `LOW_FIT_WARNING`. The checkout flow must block payment and display the reasons to the candidate.

---

## 3. Error Handling Examples

| Case                  | Input                                           | Status | Response                                                    |
|-----------------------|-------------------------------------------------|--------|-------------------------------------------------------------|
| Unknown candidate     | `{"candidate_id": 9999, "job_id": 1}`           | 404    | `{"error": "Candidate not found"}`                          |
| Unknown job           | `{"candidate_id": 1, "job_id": 9999}`           | 404    | `{"error": "Job not found"}`                                |
| Malformed body        | `{"candidate_id": "abc"}`                       | 422    | FastAPI validation error (missing job_id, invalid int)       |

---

## Full verification evidence

See [verification_log.md](verification_log.md) for complete raw terminal output of every test, API call, and evaluation run.
