# 2-Minute Live Demo Verification Script (Stage E.4)

Follow these steps to demonstrate the Cold-Start Recommendation System to evaluators:

## Step 1: Real-Data Pipeline Execution & Evidence
Run the complete pipeline runner script:
```bash
bash run_all.sh
```
- **What to highlight**: Show stdout metrics printed live:
  - Synthetic dataset creation: 500 jobs, 2000 users, 608 cold-start users, ~34k interactions.
  - Train/Serve Skew Audit: `max_diff = 0.0` (`PASSED_ZERO_SKEW`).
  - Model Performance: $P@5 = 0.9253$, $MAP = 0.9284$, $nDCG@10 = 0.9611$ vs Popularity $P@5 = 0.0484$.
  - Offline $P@5$ Lift: $+0.8769$; Expected Online Lift: $+0.4385$.
  - Fairness Audit: Parity Ratio = $0.9055$ (Exceeds EEOC $80\%$ threshold).
  - All 18 unit tests passing cleanly.

---

## Step 2: Fresh Candidate Onboarding API Request
Start the FastAPI server (or call endpoints via test harness / curl):
```bash
py -3 -m uvicorn api.serve:app --reload
```
Hit the `/recommend` endpoint with a new candidate profile:
```http
GET http://127.0.0.1:8000/recommend?user_skills=python,sql&k=5
```
- **What to highlight**: Point out the response JSON:
  - `source`: `"model"`
  - `model_version`: `"1.0.0"`
  - `explanation`: Plain-English explanation detailing overlap %, weights used, and common skills.
  - `item_details`: Exploitation vs Exploration slot breakdown (`is_exploration: true/false`).

---

## Step 3: Deliberate Outage Induction & Graceful Fallback (Failure Handling)
Induce an artificial model outage via the API:
```http
POST http://127.0.0.1:8000/simulate_outage?down=true
```
Now send the exact same recommendation request again:
```http
GET http://127.0.0.1:8000/recommend?user_skills=python,sql&k=5
```
- **What to highlight**:
  - `source`: `"popularity_fallback"`
  - Candidate receives a non-empty, top-popular job list immediately.
  - Screen is **NEVER empty** even during total model failure.

Restore operational status:
```http
POST http://127.0.0.1:8000/simulate_outage?down=false
```