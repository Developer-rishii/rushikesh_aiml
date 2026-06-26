# PlaceMux Spend-Quality Guardrail (Task 8)

This repository contains the "Spend-Quality Guardrail", an API layer that sits upstream of any paid action in PlaceMux (e.g., applying to a job) and flags low-fit matches BEFORE money moves.

## Context
PlaceMux has an existing matching pipeline (Task 6/7) that produces match scores between candidates and jobs. This guardrail layer consumes those scores to decide whether an application is "fit to spend money on", enforcing a calibrated quality threshold.

*(Note: Data dependencies `candidate_profiles.csv` and `jobs.csv` along with a baseline model mock were generated to simulate the upstream pipeline handoff as instructed).*

## Project Structure
```text
week3-task8/
  data/
    candidate_profiles.csv
    jobs.csv
    match_history.csv (500+ records)
  src/
    baseline_matcher.py
    guardrail.py
    threshold_calibration.py
    evaluate_guardrail.py
    explainability.py
    api.py
  models/
    baseline_model.pkl
  metrics/
    guardrail_metrics.json
    experiment_log.csv
  tests/
    test_guardrail.py
    test_api.py
  docs/
    architecture.md
    demo_guide.md
    handoff_notes.md
```

## Running the System

**1. Calibrate & Evaluate Threshold**
```bash
python -m src.threshold_calibration
python -m src.evaluate_guardrail
```

**2. Start the API**
```bash
uvicorn src.api:app --reload
```

**3. Run Tests**
```bash
pytest tests/ -v
```

## Live Demo Summary

The threshold was calibrated on a 70/15/15 split. The best threshold for this dataset is **40.0%**.
Performance on test data:
- Precision: 55.9%
- Recall: 59.4%
- FPR: 25.9%

**Example: A Good Match (Passes Guardrail)**
*The candidate has a 71.4% score (above 40% threshold).*
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

**Example: A Low-Fit Match (Blocked by Guardrail)**
*The candidate only has a 33.3% score.*
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

The guardrail blocks payment for the low-fit match, saving the user money and explaining exactly why (missing Java, C++).

## Handoff
Please refer to `docs/handoff_notes.md` for integration instructions for the payments team.
