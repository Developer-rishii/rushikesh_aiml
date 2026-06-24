# Demo Guide: PlaceMux Quality Baseline

This guide walks through a complete end-to-end demo using real generated data, the actual trained model, and live API output.

## Prerequisites
```bash
pip install -r requirements.txt
```

## Step 1: Generate Data and Train the Model

```bash
python src/data_loader.py      # Generates data/candidate_profiles.csv and data/jobs.csv
python src/train.py             # Trains LogisticRegression, saves to models/baseline_model.pkl
python src/evaluate.py          # Evaluates on held-out test set, saves metrics/metrics.json
```

## Step 2: Start the API

```bash
uvicorn src.api:app --port 8000
```

## Step 3: End-to-End Walkthrough

### The Candidate
**Candidate ID 10001** — "Frank 1"
- Skills: Docker, GCP, Kubernetes, SQL, Azure, Machine Learning, Python, C++
- Experience: 8 years
- Education: Bachelor's
- Certifications: 0

### The Job
**Job ID 1005** — "Data Scientist" at TechCompany_4
- Required Skills: SQL, ML
- Minimum Experience: 0 years
- Preferred Education: Master's

### Baseline Score
Using the formula `(required_skills_matched / total_required_skills) × 100`:
- SQL ✅ matched, ML ✅ matched (as "Machine Learning")
- Baseline Score: **50.0%** (1 of 2 required skills matched by exact name; "Machine Learning" ≠ "ML" in string matching)

### Model Prediction
```json
POST /match
{"candidate_id": 10001, "job_id": 1005}

Response:
{
  "candidate_id": 10001,
  "job_id": 1005,
  "baseline_score": 50.0,
  "prediction_score": 50.57,
  "prediction": 1,
  "explanation": [
    "Partial skill match (50.0% of required skills met).",
    "Experience requirement satisfied (exceeds by 8.0 years).",
    "Did not meet preferred education level."
  ]
}
```

### Explanation Breakdown
1. **Skill Match:** The candidate has SQL which matches the job's requirement, but "Machine Learning" doesn't exact-match "ML" in the baseline, so coverage is 50%.
2. **Experience:** The candidate has 8 years of experience vs. the 0-year requirement — well exceeded.
3. **Education:** The job prefers a Master's but the candidate has a Bachelor's.

### Test-Set Metrics (from `metrics/metrics.json`)
| Metric | Value |
|--------|-------|
| Precision | 0.8365 |
| Recall | 0.6971 |
| Accuracy | 0.9599 |
| F1 Score | 0.7605 |
| False Positive Rate | 0.0137 |

These are the numbers the baseline must be compared against. Any future model must improve on these metrics to justify replacing the baseline.
