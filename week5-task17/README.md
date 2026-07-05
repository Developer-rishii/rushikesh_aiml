# Task 17: Recommendation Engine v1

## Overview

This is a trained ML recommendation engine for PlaceMux that ranks jobs for students using verified skills, experience, and historical outcome data. It replaces the naive rule-based system with a fully documented Gradient Boosting approach, complete with held-out evaluation, plain-English explanations per recommendation, and strictly enforced college-level data isolation.

## What is Recommendation v1?

```text
                                 ┌───────────────┐
                                 │               │
  Student Profile ──────────────►│    Feature    │
  (Skills, Exp., College)        │  Engineering  │
                                 │               │
                                 └───────┬───────┘
                                         │
                                         ▼
                                 ┌───────────────┐
                                 │ Trained Model │
  Candidate Jobs ───────────────►│  (vs Baseline)│
                                 │               │
                                 └───────┬───────┘
                                         │
                                         ▼
                                 ┌───────────────┐
                                 │  Ranked Jobs  │
                                 │ + Explanation │
                                 └───────────────┘
```

**Example Output:**
*(Note: student_100/college_13 is a deterministic pairing because `generate_data.py` uses `seed=42`, keeping examples consistent across reruns)*
```json
{
  "job_id": "job_478",
  "score": 0.9099,
  "explanation": "You possess all the required skills for this role. Your proficiency exceeds the minimum requirements. Your years of experience perfectly align with the seniority of this role. Students from your college have historically had an average hire rate for similar roles."
}
```

## Project Structure

```text
Task17_Recommendation_v1/
├── rec_v1/
│   ├── api.py                 # FastAPI serving layer (tenant-isolated)
│   ├── baseline.py            # Rule-based ranker for performance comparison
│   ├── demo.py                # End-to-end execution, evaluation, and live walkthrough
│   ├── evaluate.py            # Global and per-segment evaluation script
│   ├── experiments.jsonl      # Hyperparameter sweep logs
│   ├── explain.py             # Logic for plain-English reasoning
│   ├── features.py            # Feature engineering pipeline
│   ├── generate_data.py       # Realistic synthetic data & edge-case injection
│   ├── model.pkl              # Persisted ML model artifact
│   ├── test_edge_cases.py     # Automated testing for failure modes & explanation consistency
│   ├── test_isolation.py      # Automated testing for data isolation
│   ├── train.py               # Model training and hyperparameter tuning
│   └── data/                  # Generated CSV datasets
├── requirements.txt           # Explicit version pins for reproducibility
```

## Model & Scoring Approach

We trained a **GradientBoostingClassifier** (via scikit-learn) to predict the likelihood of a student being hired. The model uses four core features built in `features.py`:
1. **`skill_overlap_ratio`**
2. **`proficiency_gap`** *(Normalized to [0, 1] to prevent magnitude dominance)*
3. **`experience_fit`**
4. **`college_hire_prior`**

This significantly outperforms our **Baseline**, which was a naive rule-based ranker that scored candidates *solely* on skill overlap ratio. The trained model successfully captures non-linear interactions (e.g., strong experience compensating for a partial skill match) that the baseline misses.

## Installation & Setup

Ensure you are using **Python 3.11+**.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline in order
cd rec_v1
python generate_data.py     # Generates 75k outcomes & simulated profiles
python train.py             # Trains model and runs hyperparameter sweep
python evaluate.py          # Prints baseline vs model metrics
python demo.py              # Runs the live walkthrough and isolation tests
uvicorn api:app --reload    # Start the API server
```

## Usage

### 1. Direct Python API
You can load the model and features directly via `joblib`:
```python
import joblib
from features import FeatureEngineer

artifact = joblib.load("model.pkl")
model = artifact["model"]
features_to_use = artifact["features_to_use"]
# Instantiate FeatureEngineer, load context, and run transform()
```

### 2. REST API
When the server is running, you can hit the recommendation endpoints. Note that `x-college-id` is strictly required.
```bash
curl -s -X POST http://localhost:8000/recommend \
     -H "Content-Type: application/json" \
     -H "x-college-id: college_13" \
     -d '{"student_id": "student_100"}'
```

### 3. Quick Demo Script
Run `python demo.py` to execute the complete pipeline walkthrough, print test set evaluations, generate live explanations for a student, and run the automated isolation test suite in one go.

## API Endpoints

- **`GET /health`** - Basic health check.
- **`POST /recommend`** - Given a `student_id`, returns the top 5 ranked jobs with plain-English explanations.
- **`POST /recommend/reverse`** - Given a `job_id`, returns the top 5 students *from the requesting college only*.

**Example Request (`POST /recommend`)**:
```json
{
    "student_id": "student_100"
}
```

**Example Response**:
```json
{
  "student_id": "student_100",
  "recommendations": [
    {
      "job_id": "job_478",
      "score": 0.9099,
      "explanation": "You possess all the required skills for this role. Your proficiency exceeds the minimum requirements. Your years of experience perfectly align with the seniority of this role. Students from your college have historically had an average hire rate for similar roles."
    }
  ]
}
```

## Evaluation Metrics

Performance on the **held-out test set** (11,250 candidate interactions):

| Metric       | Baseline   | ML Model   | Delta     |
|--------------|------------|------------|-----------|
| **AUC**      | 0.8663     | 0.9414     | +0.0751   |
| **Precision**| 0.6217     | 0.6396     | +0.0179   |
| **Recall**   | 0.5608     | 0.4941     | -0.0667   |
| **FPR**      | 0.0079     | 0.0065     | -0.0015   |

*(Note: Normalizing the `proficiency_gap` did not fundamentally change the AUC from our prior run, but it correctly stabilizes the feature magnitude for Gradient Boosting).*

### Segment Breakdown (AUC)

**By Top Colleges:**
- `college_12`: Baseline 0.867 | Model **0.927**
- `college_7`: Baseline 0.948 | Model **0.977**
- `college_15`: Baseline 0.765 | Model **0.880**

**By Seniority Level:**
- Level 0: Baseline 0.817 | Model **0.904**
- Level 1: Baseline 0.885 | Model **0.941**
- Level 2: Baseline 0.719 | Model **0.875**
- Level 3: Baseline 0.889 | Model **0.961**
- Level 4: Baseline 0.917 | Model **0.962**
- Level 5: Baseline 0.959 | Model **0.962**

## Explainability

Every recommendation includes a plain-English explanation generated by `explain.py`. It translates raw feature values (skill overlap, proficiency gaps, experience fit, college placement priors) into readable reasoning. 

Furthermore, `test_edge_cases.py` explicitly tests for **explanation-ranking contradictions**. If a job outranks another job despite having worse core skills, the explanation is strictly enforced to state exactly why (e.g., college placement outcome priors pulled it up).

## Data Isolation & Safety

Tenant-level isolation is strictly enforced. Students and jobs are securely siloed by `college_id`. 
This is actively validated via `test_isolation.py`, which proves:
1. A request from `college_A` attempting to query a student from `college_B` is strictly rejected with a `403 Forbidden`.
2. Reverse recommending (finding students for a job) only returns candidates belonging to the requesting `x-college-id`.

## Failure Modes Handled

The system handles common edge cases gracefully without crashing, validated by `test_edge_cases.py`:
- **Cold-Start Student**: A student with zero verified skills receives an empty recommendation list and a polite prompt to update their profile (Status 200).
- **Unknown Student/Job**: Querying non-existent IDs returns a `404 Not Found`.
- **Malformed Request**: Missing fields result in a `422 Unprocessable Entity`.
- **Duplicate Skills**: Resolved internally by taking the maximum proficiency score for duplicate skill entries.
- **Zero Requirement Jobs**: Safely handled by yielding a perfect overlap ratio internally rather than dividing by zero.

## Design Decisions

- **Trained Model vs Rules:** The baseline relies solely on skill overlap. A trained Gradient Boosting model successfully weights multiple variables (like experience fit and proficiency gaps) dynamically, yielding superior AUC and precision.
- **Held-Out Evaluation:** Ensures our performance metrics reflect generalized accuracy on unseen applications, avoiding overfitting.
- **Mandatory Explainability:** In a hiring context, black-box decisions are unacceptable. By mapping all 4 features back to readable explanations, candidates trust the system and understand how to improve their profiles.

## Sample Data

Generated realistic synthetic data for testing and training:
- **Total Colleges:** 20
- **Total Students:** 5,000
- **Total Companies:** 80
- **Total Jobs:** 500
- **Simulated Applications (Train/Test Signal):** 75,000

## Live Verification Transcript

*Date: 2026-07-05*

```
> python generate_data.py
=== Generating Synthetic Data ===
Generating 5000 students across 20 colleges...
Generating student skills...
Generating 500 jobs across 80 companies...
Generating job requirements...
Simulating historical outcomes (was_shortlisted, was_hired)...
=== Dataset Generation Complete ===
Students generated: 5000
Jobs generated: 500
Student Skills entries: 99402 (Missing prof: 3991)
Job Skills entries: 4816
Outcomes (applications): 75000
  - Shortlisted: 4215 (5.6%)
  - Hired: 1701 (2.3%)

> python train.py
=== Training Pipeline ===
Data Split: Train=52500, Val=11250, Test=11250
Engineering features...
Running Hyperparameter Sweep...
Run 0: AUC=0.9466, Prec=0.7065 | {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}
Run 1: AUC=0.9467, Prec=0.7056 | {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}
Run 2: AUC=0.9453, Prec=0.7095 | {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.05}
Run 3: AUC=0.9464, Prec=0.6973 | {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05}
Run 4: AUC=0.9388, Prec=0.6332 | {'n_estimators': 50, 'max_depth': 5, 'learning_rate': 0.2}

Best Model AUC: 0.9467 with params: {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}
Saved best model to model.pkl

> python evaluate.py
=== Evaluation (Held-out Test Set) ===

Test Set: 11250 candidate interactions

--- Global Metrics ---
Metric       | Baseline   | ML Model   | Delta     
--------------------------------------------------
AUC          | 0.8663     | 0.9414     | +0.0751
Precision    | 0.6217     | 0.6396     | +0.0179
Recall       | 0.5608     | 0.4941     | -0.0667
FPR          | 0.0079     | 0.0065     | -0.0015

> pytest test_isolation.py test_edge_cases.py
============================= test session starts =============================
collected 7 items
test_isolation.py::test_tenant_isolation PASSED                          [ 14%]
test_isolation.py::test_reverse_recommend_isolation PASSED               [ 28%]
test_edge_cases.py::test_unknown_student PASSED                          [ 42%]
test_edge_cases.py::test_malformed_request PASSED                        [ 57%]
test_edge_cases.py::test_cold_start_student PASSED                       [ 71%]
test_edge_cases.py::test_unknown_job_reverse_recommend PASSED            [ 85%]
test_edge_cases.py::test_explanation_consistency PASSED                  [100%]
======================== 7 passed, 2 warnings in 5.35s ========================

> python -c "import requests, json; print(json.dumps(requests.post('http://localhost:8000/recommend', headers={'x-college-id': 'college_13'}, json={'student_id': 'student_100'}).json(), indent=2))"
{
  "student_id": "student_100",
  "recommendations": [
    {
      "job_id": "job_478",
      "score": 0.9099222063093553,
      "explanation": "You possess all the required skills for this role. Your proficiency exceeds the minimum requirements. Your years of experience perfectly align with the seniority of this role. Students from your college have historically had an average hire rate for similar roles."
    },
    {
      "job_id": "job_308",
      "score": 0.9099222063093553,
      "explanation": "You possess all the required skills for this role. Your proficiency exceeds the minimum requirements. Your years of experience perfectly align with the seniority of this role. Students from your college have historically had an average hire rate for similar roles."
    }
  ]
}
```

## Definition of Done

- [x] **Core Deliverable:** End-to-end ML training pipeline and inference API created.
- [x] **Real-Data Quality:** Realistic dataset with scale and edge cases generated.
- [x] **Live Verification:** Model formally evaluated against a baseline.
- [x] **Safety & Resilience:** Edge cases handled gracefully, consistency checks in place, and data isolation enforced via Pytest.

## Troubleshooting

- **`ModuleNotFoundError` or similar**: Ensure you ran `pip install -r requirements.txt`.
- **`FileNotFoundError: model.pkl`**: You need to run `python train.py` first to generate the persisted model artifact.
- **403 Forbidden on API Requests**: Verify you are providing a valid `x-college-id` header that actually matches the requested student's enrolled college.

## Author / License
`PlaceMux AI/ML Team · Phase 2 · Week 5 · Task 17`
*Internal use only. PlaceMux confidential.*
