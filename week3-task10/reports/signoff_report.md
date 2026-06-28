# PlaceMux Quality Sign-Off Report

**Generated:** 2026-06-28 14:52:37 UTC
**Test Set Size:** 115 samples (held-out, never trained/tuned on)

---

## 1. What "Good" Looks Like

This sign-off verifies that the existing matching/recommendation system **still works
correctly** after monetization integration.

---

## 2. Baseline Definition & Numbers

**Baseline rule:** Rank candidates by raw overlap count of (required skills & verified skills).

### Overall Baseline Performance (Test Set)

|                      | Precision |    Recall |       FPR |     N |
|----------------------|-----------|-----------|-----------|-------|
| Baseline             |     0.134 |       1.0 |    0.8235 |   115 |

---

## 3. Trained ML Model

### Labeling Rule (Auditable)

```
label = 1 (good match) IF AND ONLY IF:
  1. Verified-skill coverage of required skills >= 80%
  2. No required skill is missing by more than 1 level
     (for every skill the student HAS: student_level >= req_level - 1)
label = 0 otherwise
```

### Model Details

| Parameter            | Value                             |
|----------------------|-----------------------------------|
| Algorithm            | RandomForestClassifier            |
| n_estimators         | 100 |
| max_depth            | 8 |
| Train / Val / Test   | 345 / 115 / 115 |
| Train accuracy       | 0.9913 |
| Val P / R / F1       | 0.8125 / 1.0 / 0.8966 |
| Test P / R / F1      | 0.6875 / 0.8462 / 0.7586 |
| Training time        | 0.09s |

### Feature Importances

| Feature                        | Importance |
|--------------------------------|------------|
| weighted_coverage_score        |     0.4666 |
| min_level_delta                |     0.1902 |
| num_missing_required           |     0.1444 |
| mean_level_delta               |     0.0793 |
| skill_overlap_count            |     0.0403 |
| max_level_delta                |     0.0351 |
| years_of_exposure              |     0.0240 |
| price_tier_premium             |     0.0084 |
| pay_failed                     |     0.0048 |
| price_tier_free                |     0.0034 |
| pay_refunded                   |     0.0011 |
| price_tier_basic               |     0.0010 |
| pay_success                    |     0.0010 |
| pay_pending                    |     0.0003 |

---

## 4. Model vs Baseline - Pre/Post Monetization

### Overall (Test Set)

|                      | Precision |    Recall |       FPR |     N |
|----------------------|-----------|-----------|-----------|-------|
| Baseline             |     0.134 |       1.0 |    0.8235 |   115 |
| Trained model        |    0.6875 |    0.8462 |     0.049 |   115 |
| **Delta (mod-base)** |   +0.5535 |   -0.1538 |   -0.7745 |       |

### Pre-Monetization (Free Applications)

|                      | Precision |    Recall |       FPR |     N |
|----------------------|-----------|-----------|-----------|-------|
| Baseline             |    0.1212 |       1.0 |    0.8056 |    40 |
| Trained model        |    0.5714 |       1.0 |    0.0833 |    40 |
| **Delta (mod-base)** |   +0.4502 |      +0.0 |   -0.7223 |       |

### Post-Monetization (Paid Applications)

|                      | Precision |    Recall |       FPR |     N |
|----------------------|-----------|-----------|-----------|-------|
| Baseline             |    0.1406 |       1.0 |    0.8333 |    75 |
| Trained model        |    0.7778 |    0.7778 |    0.0303 |    75 |
| **Delta (mod-base)** |   +0.6372 |   -0.2222 |    -0.803 |       |

### Guardrail: Post-Monetization Degradation Check

- **Precision degraded?** [OK] NO (pre=0.5714, post=0.7778)
- **Recall degraded?** [!] YES (pre=1.0, post=0.7778)
- **FPR increased degraded?** [OK] NO (pre=0.0833, post=0.0303)

### Model vs Baseline Verdict

[OK] Trained model **beats baseline** on: precision, lower FPR.

Overall delta: precision +0.5535, recall -0.1538, FPR -0.7745.

### Segment Breakdowns


**By Price Tier:**

|                      | Precision |    Recall |       FPR |     N |
|----------------------|-----------|-----------|-----------|-------|
| free baseline        |    0.1212 |       1.0 |    0.8056 |    40 |
| free model           |    0.5714 |       1.0 |    0.0833 |    40 |
| **Delta (mod-base)** |   +0.4502 |      +0.0 |   -0.7223 |       |
|----------------------|-----------|-----------|-----------|-------|
| basic baseline       |      0.12 |       1.0 |    0.9167 |    27 |
| basic model          |       0.5 |    0.3333 |    0.0417 |    27 |
| **Delta (mod-base)** |     +0.38 |   -0.6667 |    -0.875 |       |
|----------------------|-----------|-----------|-----------|-------|
| premium baseline     |    0.1538 |       1.0 |    0.7857 |    48 |
| premium model        |    0.8571 |       1.0 |    0.0238 |    48 |
| **Delta (mod-base)** |   +0.7033 |      +0.0 |   -0.7619 |       |
|----------------------|-----------|-----------|-----------|-------|


**By Payment Status:**

|                      | Precision |    Recall |       FPR |     N |
|----------------------|-----------|-----------|-----------|-------|
| success baseline     |    0.1493 |       1.0 |    0.7808 |    83 |
| success model        |    0.6667 |       0.8 |    0.0548 |    83 |
| **Delta (mod-base)** |   +0.5174 |      -0.2 |    -0.726 |       |
|----------------------|-----------|-----------|-----------|-------|
| failed baseline      |    0.0769 |       1.0 |    0.8571 |    15 |
| failed model         |       1.0 |       1.0 |       0.0 |    15 |
| **Delta (mod-base)** |   +0.9231 |      +0.0 |   -0.8571 |       |
|----------------------|-----------|-----------|-----------|-------|
| pending baseline     |    0.0714 |       1.0 |       1.0 |    14 |
| pending model        |       0.5 |       1.0 |    0.0769 |    14 |
| **Delta (mod-base)** |   +0.4286 |      +0.0 |   -0.9231 |       |
|----------------------|-----------|-----------|-----------|-------|
| refunded baseline    |    0.3333 |       1.0 |       1.0 |     3 |
| refunded model       |       1.0 |       1.0 |       0.0 |     3 |
| **Delta (mod-base)** |   +0.6667 |      +0.0 |      -1.0 |       |
|----------------------|-----------|-----------|-----------|-------|

---

## 5. Worked Example (Explainability Walkthrough)

Use the live API to get a real per-prediction explanation:

```
GET /match/S010/J005
```

---

## 6. Edge Cases Tested

| Edge Case                        | Handled By                          | Test                                |
|----------------------------------|-------------------------------------|-------------------------------------|
| Payment fails mid-application    | `reconciliation.handle_payment_failure()` | `test_student_retains_application_on_failure` |
| Student charged without match    | `reconciliation.reconcile_payments()` | `test_charged_without_match_flagged` |
| Gateway/recorded amount mismatch | `reconciliation.validate_amounts()` | `test_mismatch_detected` |
| Duplicate/partial payments       | `reconciliation.reconcile_payments()` | `test_duplicates_detected` |
| Missing skill scores (NaN)       | `features.build_features()`         | `test_baseline_handles_nan_skills` |
| Zero-overlap JD                  | Handled gracefully                  | `test_baseline_zero_overlap` |

---

## 7. Self-Check Questions

### Can "Quality Sign-Off" be shown working live, not just described?

**YES.** Start the API with `uvicorn api.main:app --port 8000` and hit:
- `/match/S010/J005` for a live prediction with explanation
- `/signoff/report` for the full metrics JSON
- `/signoff/reconciliation` for live mismatch detection

### What happens if a payment fails halfway?

**The student retains their application and is never charged without a match record.**
See `src/reconciliation.py::handle_payment_failure()`. If `gateway_amount > 0` and
`payment_status == "failed"`, a refund is initiated automatically. The application
status is set to `payment_failed`, NOT deleted.

### How do we know our records match what the gateway collected?

**`reconciliation.validate_amounts()` compares every `gateway_amount` to `recorded_amount`
with a configurable tolerance.** Mismatches are flagged with severity.
The `/signoff/reconciliation` endpoint returns all flagged discrepancies live.

### Are we in real-money/test mode?

**This is built and validated in test mode on synthetic data.** The payment amounts
($0, $29.99, $99.99) are realistic but generated.
