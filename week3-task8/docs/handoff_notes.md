# Spend Protection Handoff Notes

This document is for the Payments/Receipts team to integrate the spend-quality guardrail into the checkout flow.

## 1. The API Contract
Before a student can pay the ₹100 application fee, you must call the guardrail check endpoint.

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
  "threshold_used": 40.0,
  "reason": [
    "Match score 35.5% is below the 40.0% spend-safety threshold.",
    "Missing required skills: Python, AWS"
  ]
}
```

## 2. What the Status Means
- `OK`: The candidate has a decent chance of being shortlisted. **Proceed to the payment gateway.**
- `LOW_FIT_WARNING`: The candidate is highly unlikely to be shortlisted. **Do NOT authorize spend.** Instead, show the warning message and the `reason` array to the user to explain why their payment was blocked.

## 3. Current Performance
Based on our calibration dataset (4000+ historical evaluations):
- **Precision**: ~55-65% (When we warn, the match was genuinely bad)
- **Recall**: ~60% (We catch 60% of all bad matches)
- **FPR**: ~25% (We wrongly block 25% of good matches - a conservative threshold)
*(See `metrics/guardrail_metrics.json` for exact latest numbers).*

## 4. Re-Calibrating the Threshold
If match quality drifts or business requirements change (e.g., we want to block fewer people):
1. Add new data to `data/match_history.csv`.
2. Run `python -m src.threshold_calibration`.
3. Restart the FastAPI service to load the new threshold.

## 5. Out of Scope
This service **DOES NOT**:
- Issue receipts
- Process refunds
- Talk to Stripe, Razorpay, or any payment gateway
- Deduct money
It only acts as a traffic light (Green/Red) for the payments flow.
