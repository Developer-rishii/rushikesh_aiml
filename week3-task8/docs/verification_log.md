# Verification Log — Spend-Quality Guardrail

All outputs below are **real terminal outputs** captured by running the actual scripts and hitting the actual API. Nothing is mocked or assumed.

---

## 1. Real Model Training (Fix 1)

**Command**: `python src/train_baseline.py`

**Output**:
```
Model trained and saved. Real match history generated.

--- 5 Real Candidate-Job Pairs ---
Cand 28 & Job 39:
  Baseline Score: 0.0% (Skill overlap)
  Prediction Score: 3.64% (LogisticRegression)
  Is Success: 0

Cand 69 & Job 40:
  Baseline Score: 33.33% (Skill overlap)
  Prediction Score: 39.29% (LogisticRegression)
  Is Success: 0

Cand 188 & Job 82:
  Baseline Score: 20.0% (Skill overlap)
  Prediction Score: 8.3% (LogisticRegression)
  Is Success: 0

Cand 31 & Job 120:
  Baseline Score: 0.0% (Skill overlap)
  Prediction Score: 2.72% (LogisticRegression)
  Is Success: 0

Cand 245 & Job 72:
  Baseline Score: 42.86% (Skill overlap)
  Prediction Score: 54.73% (LogisticRegression)
  Is Success: 0
```

The model is a real `scikit-learn LogisticRegression` trained on 4 engineered features: `skill_overlap_percentage`, `experience_gap`, `education_match`, `certification_match_count`. The `prediction_score` is the model's `predict_proba` output scaled to 0-100.

---

## 2. Threshold Calibration & Evaluation (Fix 2)

**Command**: `python -m src.evaluate_guardrail`

**Output**:
```
Train+Val size: 1700, Test size: 300

--- CALIBRATION (on train+val) ---
=================================================================
  THRESHOLD CALIBRATION RESULTS
=================================================================
  Best threshold (max F1): 55%
  Best F1: 0.8816

  Metric                      Dumb Baseline      Calibrated
  ------------------------- --------------- ---------------
  Precision                          0.6912          0.9095
  Recall                             1.0000          0.8553
  Accuracy                           0.6912          0.8412
  F1 Score                           0.8174          0.8816
  False Positive Rate                1.0000          0.1905
=================================================================

  Business rationale: F1 maximization balances blocking bad matches
  (recall) against not wrongly blocking good matches (precision).
  The calibrated threshold improves precision by
  +21.8pp over the dumb baseline while maintaining
  85.5% recall.
=================================================================

=================================================================
  FINAL EVALUATION ON HELD-OUT TEST SET
=================================================================
  Threshold used: 55%

  Metric                      Dumb Baseline      Calibrated
  ------------------------- --------------- ---------------
  Precision                          0.7200          0.8966
  Recall                             1.0000          0.8426
  Accuracy                           0.7200          0.8167
  F1 Score                           0.8372          0.8687
  False Positive Rate                1.0000          0.2500
=================================================================

Metrics saved to metrics/guardrail_metrics.json
{
    "precision": 0.8966,
    "recall": 0.8426,
    "accuracy": 0.8167,
    "f1_score": 0.8687,
    "false_positive_rate": 0.25,
    "threshold_used": 55,
    "dumb_baseline": {
        "precision": 0.72,
        "recall": 1.0,
        "accuracy": 0.72,
        "f1_score": 0.8372,
        "false_positive_rate": 1.0
    }
}
```

**Key improvement**: Precision went from 72.0% (dumb baseline) to **89.7%** (calibrated). FPR dropped from 100% to **25.0%**.

---

## 3. Full Test Suite (Fix 3a)

**Command**: `python -m pytest tests/ -v`

**Output**:
```
============================= test session starts =============================
platform win32 -- Python 3.13.4, pytest-9.0.3, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: D:\Placemux-aiml\week3-task8
plugins: anyio-4.11.0, mock-3.15.1
collecting ... collected 8 items

tests/test_api.py::test_unknown_candidate PASSED                         [ 12%]
tests/test_api.py::test_unknown_job PASSED                               [ 25%]
tests/test_api.py::test_malformed_request PASSED                         [ 37%]
tests/test_api.py::test_successful_request PASSED                        [ 50%]
tests/test_guardrail.py::test_zero_skill_overlap PASSED                  [ 62%]
tests/test_guardrail.py::test_good_match PASSED                          [ 75%]
tests/test_guardrail.py::test_calibration_small_dataset PASSED           [ 87%]
tests/test_guardrail.py::test_calibration_one_class PASSED               [100%]

======================== 8 passed, 1 warning in 1.47s =========================
```

All 8 tests pass.

---

## 4. Live API — Good Match (Fix 3b)

**Request**: `POST /guardrail-check` with `{"candidate_id": 1, "job_id": 19}`

**Response** (status 200):
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

---

## 5. Live API — Low-Fit Warning (Fix 3b)

**Request**: `POST /guardrail-check` with `{"candidate_id": 1, "job_id": 1}`

**Response** (status 200):
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

---

## 6. Failure Case 1 — Unknown Candidate

**Request**: `POST /guardrail-check` with `{"candidate_id": 9999, "job_id": 1}`

**Response** (status 404):
```json
{
  "error": "Candidate not found"
}
```

---

## 7. Failure Case 2 — Unknown Job

**Request**: `POST /guardrail-check` with `{"candidate_id": 1, "job_id": 9999}`

**Response** (status 404):
```json
{
  "error": "Job not found"
}
```

---

## 8. Failure Case 3 — Malformed Request

**Request**: `POST /guardrail-check` with `{"candidate_id": "abc"}`

**Response** (status 422):
```json
{
  "detail": [
    {
      "type": "int_parsing",
      "loc": ["body", "candidate_id"],
      "msg": "Input should be a valid integer, unable to parse string as an integer",
      "input": "abc"
    },
    {
      "type": "missing",
      "loc": ["body", "job_id"],
      "msg": "Field required",
      "input": {"candidate_id": "abc"}
    }
  ]
}
```

---

## 9. Experiment Log (last 10 rows)

```
  run_id  timestamp  threshold  precision  recall  accuracy  f1_score  false_positive_rate
a2315c03 1782489432         86     0.7365  0.9660    0.7376    0.8358               0.7733
a2315c03 1782489432         87     0.7326  0.9745    0.7365    0.8364               0.7962
a2315c03 1782489432         88     0.7275  0.9770    0.7312    0.8340               0.8190
a2315c03 1782489432         89     0.7224  0.9787    0.7253    0.8312               0.8419
a2315c03 1782489432         90     0.7172  0.9821    0.7200    0.8290               0.8667
a2315c03 1782489432         91     0.7145  0.9838    0.7171    0.8278               0.8800
a2315c03 1782489432         92     0.7102  0.9864    0.7124    0.8258               0.9010
a2315c03 1782489432         93     0.7066  0.9881    0.7082    0.8240               0.9181
a2315c03 1782489432         94     0.7029  0.9906    0.7041    0.8223               0.9371
a2315c03 1782489432         95     0.6997  0.9915    0.7000    0.8204               0.9524
```

Total rows in experiment log: 91 (thresholds swept from 5% to 95% in 1% increments).
