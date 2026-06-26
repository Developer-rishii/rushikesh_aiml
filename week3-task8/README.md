# PlaceMux Spend-Quality Guardrail (Task 8)

This repository implements a **Spend-Quality Guardrail** — an API layer that sits upstream of any paid action in PlaceMux and flags low-fit matches BEFORE money moves.

## Upstream Dependency

The guardrail relies on an upstream matching pipeline. Since the Task-6 artifacts were not directly available in this workspace, a **real equivalent pipeline was built from scratch**:

- A `scikit-learn LogisticRegression` model was trained on 2000 candidate-job pairs using 4 engineered features: `skill_overlap_percentage`, `experience_gap`, `education_match`, `certification_match_count`.
- The model produces real `prediction_score` values (0-100) via `predict_proba`, not random/mocked scores.
- The trained model is saved as `models/baseline_model.pkl`.

## Project Structure

```text
week3-task8/
  data/
    candidate_profiles.csv    (300 candidates)
    jobs.csv                  (120 jobs)
    match_history.csv         (2000 labeled candidate-job records)
  src/
    train_baseline.py         (trains the real LogisticRegression model)
    baseline_matcher.py       (computes features + model inference)
    guardrail.py              (evaluate_guardrail function)
    threshold_calibration.py  (sweep thresholds, log to experiment_log.csv)
    evaluate_guardrail.py     (70/15/15 split evaluation)
    explainability.py         (plain-English reason generation)
    api.py                    (FastAPI service)
  models/
    baseline_model.pkl        (real trained LogisticRegression)
  metrics/
    guardrail_metrics.json    (final test-set metrics)
    experiment_log.csv        (91 threshold sweep rows)
  tests/
    test_guardrail.py
    test_api.py
  docs/
    architecture.md
    demo_guide.md
    handoff_notes.md
    verification_log.md       (all raw evidence)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the real model (generates data + model + match_history)
python src/train_baseline.py

# 3. Calibrate the threshold
python -m src.threshold_calibration

# 4. Evaluate on held-out test data
python -m src.evaluate_guardrail

# 5. Start the API
uvicorn src.api:app --reload

# 6. Run the test suite
python -m pytest tests/ -v
```

## Verified Evidence

All claims in this project are backed by real, executed output captured in [`docs/verification_log.md`](docs/verification_log.md).

### Final Metrics (held-out 15% test set, 300 records)

| Metric              | Dumb Baseline (warn everyone) | Calibrated Guardrail (threshold=55%) |
|---------------------|-------------------------------|--------------------------------------|
| Precision           | 0.7200                        | **0.8966**                           |
| Recall              | 1.0000                        | 0.8426                               |
| Accuracy            | 0.7200                        | **0.8167**                           |
| F1 Score            | 0.8372                        | **0.8687**                           |
| False Positive Rate | 1.0000                        | **0.2500**                           |

### One Good Match, One Low-Fit Match

**Good match** (candidate 1 → job 19): Score **72.03%**, status **OK**.
Reasons: "Matched skills: C++, Git, SQL", "Missing: Machine Learning, React".

**Low-fit match** (candidate 1 → job 1): Score **16.88%**, status **LOW_FIT_WARNING**.
Reasons: "Score 16.9% is below the 55.0% threshold", "Missing: AWS, Docker, Machine Learning", "No matching certifications".

### Downstream Handoff

This guardrail hands off a `fit_status` signal (`OK` or `LOW_FIT_WARNING`) to the Spend Protection / Payments team. It does NOT issue receipts, process refunds, or interact with any payment gateway. See [`docs/handoff_notes.md`](docs/handoff_notes.md) for integration instructions.
