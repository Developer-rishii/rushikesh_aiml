# Spend Protection Handoff Notes

This document is for the Payments/Receipts team to integrate the spend-quality guardrail into the checkout flow.

## 1. Upstream Dependency

The guardrail consumes a **real scikit-learn LogisticRegression model** trained on 4 engineered features derived from candidate-job pairs:

| Feature                    | Description                                    |
|----------------------------|------------------------------------------------|
| `skill_overlap_percentage` | % of required job skills the candidate has     |
| `experience_gap`           | Candidate years − Job min years (can be negative) |
| `education_match`          | 1 if candidate education matches job requirement |
| `certification_match_count`| Count of matching certifications               |

The model was trained on 2000 candidate-job pairs with realistic ground-truth labels (success determined by feature quality + noise). This is NOT a mock — the `models/baseline_model.pkl` file contains the real trained model.

## 2. The API Contract

Before a student can pay the application fee, call the guardrail check endpoint.

**Endpoint**: `POST /guardrail-check`

**Request**:
```json
{
  "candidate_id": 123,
  "job_id": 456
}
```

**Response**:
```json
{
  "candidate_id": 123,
  "job_id": 456,
  "match_score": 35.5,
  "fit_status": "LOW_FIT_WARNING",
  "threshold_used": 55,
  "reason": [
    "Match score 35.5% is below the 55.0% spend-safety threshold.",
    "Missing required skills: Python, AWS",
    "Experience gap of 2.0 years: you have 1.0 years, job requires 3.0 years.",
    "No matching certifications with the job requirements."
  ]
}
```

## 3. What the Status Means

- **`OK`**: The candidate has a reasonable chance of being shortlisted. **Proceed to the payment gateway.**
- **`LOW_FIT_WARNING`**: The candidate is unlikely to be shortlisted. **Do NOT authorize spend.** Show the warning message and the `reason` array to explain why.

## 4. Current Performance (Real, Verified)

Evaluated on a held-out 15% test set (300 records), calibrated threshold = **55%**:

| Metric              | Dumb Baseline (warn everyone) | Calibrated Guardrail |
|---------------------|-------------------------------|----------------------|
| Precision           | 0.7200                        | **0.8966**           |
| Recall              | 1.0000                        | 0.8426               |
| Accuracy            | 0.7200                        | **0.8167**           |
| F1 Score            | 0.8372                        | **0.8687**           |
| False Positive Rate | 1.0000                        | **0.2500**           |

The calibrated guardrail improves precision by **+17.7pp** over the naive "always warn" baseline, meaning when we flag a match as low-fit, we are correct **89.7%** of the time. We catch **84.3%** of genuinely bad matches.

## 5. Re-Calibrating the Threshold

If match quality drifts or business requirements change:
1. Add new labeled data to `data/match_history.csv`.
2. Run `python -m src.threshold_calibration` — this sweeps every 1% from 5-95 and logs all results to `metrics/experiment_log.csv`.
3. Run `python -m src.evaluate_guardrail` to evaluate on held-out data and update `metrics/guardrail_metrics.json`.
4. Restart the FastAPI service to load the new threshold.

## 6. Out of Scope

This service **DOES NOT**:
- Issue receipts
- Process refunds
- Talk to Stripe, Razorpay, or any payment gateway
- Deduct money

It only produces a fit/warning signal for the payments flow to consult.
