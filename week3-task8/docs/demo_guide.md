# Spend-Quality Guardrail — Live Demo Guide

This guide walks you through testing the guardrail API live to verify it properly gates spend on job applications based on the upstream match data.

## Current Calibrated Performance
*From `metrics/guardrail_metrics.json`*
- **Precision**: 55.9% (When we warn, the match was genuinely bad)
- **Recall**: 59.4% (We catch 59.4% of all bad matches)
- **Accuracy**: 68.9%
- **False Positive Rate (FPR)**: 25.9% (We wrongly block 25.9% of good matches - a conservative threshold)
- **Threshold Used**: 40.0%

---

## 1. Good Match Walkthrough (Passes Guardrail)
A candidate and job that share significant skills. 

**Request**:
```bash
curl -X POST http://127.0.0.1:8000/guardrail-check -H "Content-Type: application/json" -d '{"candidate_id": 1, "job_id": 4}'
```

**Response**:
```json
{
  "candidate_id": 1,
  "job_id": 4,
  "match_score": 71.42857142857143,
  "fit_status": "OK",
  "threshold_used": 40.0,
  "reason": [
    "Match score 71.4% passes the 40.0% spend-safety threshold.",
    "Missing required skills: Python, React"
  ]
}
```
**Outcome**: Status is `OK`. The checkout flow should authorize the application spend.

---

## 2. Bad Match Walkthrough (Fails Guardrail)
A candidate attempting to apply for a job where they lack crucial required skills.

**Request**:
```bash
curl -X POST http://127.0.0.1:8000/guardrail-check -H "Content-Type: application/json" -d '{"candidate_id": 1, "job_id": 2}'
```

**Response**:
```json
{
  "candidate_id": 1,
  "job_id": 2,
  "match_score": 33.33333333333333,
  "fit_status": "LOW_FIT_WARNING",
  "threshold_used": 40.0,
  "reason": [
    "Match score 33.3% is below the 40.0% spend-safety threshold.",
    "Missing required skills: Java, C++"
  ]
}
```
**Outcome**: Status is `LOW_FIT_WARNING`. The checkout flow must block payment and display the reasons to the candidate (e.g. they are missing Java and C++).
