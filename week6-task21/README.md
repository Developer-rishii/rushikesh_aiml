# Task 21 — DPDP Consent & Security Foundations (Fairness Audit)
**AI/ML Engineer deliverable · Week 6 Phase 3 · PlaceMux · Altrodav Technologies**

## Overview
This task implements the intelligence layer of PlaceMux for the team-wide DPDP Consent & Security Foundations day. Specifically, this repository slice focuses on the **Fairness Audit**. It evaluates the recommendation engine to ensure equitable treatment of students across protected attributes (e.g., `college_tier` and `region`), measures disparate impact, and provides a trained ML bias classifier to detect biased outcomes.

---

## How to Run the Pipeline

### 1. Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

### 2. Generate Data
First, generate the synthetic recommendation data (with deliberate bias injections to test the audit tools):
```bash
python data/generate_data.py
```

### 3. Train ML Classifier
Train the bias classifier on admin-reviewed fairness labels:
```bash
python src/bias_classifier.py
```
*Saves the trained model to `src/models/bias_classifier.pkl`.*

### 4. Run Fairness Audit
Execute the rule-based metrics and ML evaluations:
```bash
python src/evaluate.py
```
*Generates `reports/fairness_report.json` and `reports/signoff_report.md`.*

### 5. Run Tests
Ensure all edge cases and dependencies are covered:
```bash
pytest tests/ -v
```
*(All 13/13 tests must pass.)*

### 6. Serve the API
Start the FastAPI application to view live reports and assess individual bias records:
```bash
python -m uvicorn api.app:app --reload --port 8002
```

---

## API Endpoints

Once the API is running at `http://127.0.0.1:8002`, the following endpoints are available:

- **`GET /docs`** — Swagger UI to test endpoints interactively.
- **`GET /audit/report`** — Full fairness audit report JSON (baseline + ML metrics).
- **`GET /audit/signoff`** — Formal sign-off verdict.
- **`GET /audit/edge-cases`** — Demonstration of how edge cases are handled.
- **`GET /audit/student/{student_id}`** — Bias risk assessment for a specific student.
- **`GET /audit/group/{protected_attr}`** — Group-level metrics (pass `college_tier` or `region`).

---

## Edge Cases Covered
The system is tested extensively against edge cases (see `tests/test_fairness.py`):
1. **Malformed Input**: Validates missing required columns or empty dataframes (throws `ValueError`).
2. **Small Samples**: Requires ≥ 10 students for robust group-level metrics.
3. **Single-group Data**: Handles datasets with no disparity gracefully (DI defaults to 1.0 without crashing).
4. **Unknown Inference**: Returns safe error dictionaries for non-existent `student_id` API requests.
5. **Perfect Bias Detection**: Properly detects Disparate Impact (DI) = 0.0 without divide-by-zero crashes.
6. **Bias Detection Verification**: Ensures `Tier3 rural` students consistently score a higher bias risk than `Tier1 urban` students in synthetic data.

---

## Deliverable Scorecard / Definition of Done

- **Core Deliverable**: "Fairness audit (start)" built, working, and exposed via FastAPI.
- **Real-data Standard**: Handles 12,000 synthetic pairs; capable of absorbing real production DB outputs.
- **Explainability**: Every bias score includes a plain-English `reason` with top contributing features.
- **Generalization**: Model uses strict Train/Val/Test split by `student_id` to prevent data leakage.
- **Live Verification**: Verdicts, edge-cases, and metrics exposed directly through the API.
- **Dependencies**: Resolves all Unicode/Charmap OS-level bugs during logging and report saving.

---

## Hand-off & Next Steps
- **Hand-off**: Bias findings are stored in `reports/fairness_report.json` and available via the `/audit/report` endpoint. 
- **Guardrail**: Pipeline should be re-run on a cron-schedule or after every 1,000 new students are onboarded.
- **Before Launch**: 
  1. Replace `data/generate_data.py` output with real production data.
  2. Expand `fairness_labels.csv` to ≥ 500 *genuine* admin reviews.
  3. Wire the audit API to auto-run on schedule. 
*(Note: DPDP user data deletion and consent flows are managed by the Backend/Security teams this week.)*
