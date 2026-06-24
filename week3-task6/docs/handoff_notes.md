# Handoff Notes: Week-3 Monitoring Team

## Model Artifact
- **Location:** `models/baseline_model.pkl`
- **Contents:** A pickled dictionary with keys `"model"` (a scikit-learn `LogisticRegression` instance) and `"scaler"` (a `StandardScaler` instance).
- **To reload:**
  ```python
  import pickle
  with open("models/baseline_model.pkl", "rb") as f:
      artifact = pickle.load(f)
      model = artifact["model"]
      scaler = artifact["scaler"]
  ```
- If the file is missing or corrupted, the system will raise a descriptive `ValueError` — it will not silently crash.

## How to Re-Run Evaluation on New Data
1. Place updated CSVs in `data/candidate_profiles.csv` and `data/jobs.csv`.
2. Run:
   ```bash
   python src/evaluate.py
   ```
3. New metrics are written to `metrics/metrics.json` and appended to `metrics/experiment_log.csv`.

## Current Baseline Numbers
These are the numbers from the current trained model, evaluated on the held-out 15% test split:

| Metric | Value |
|--------|-------|
| Precision | 0.8365 |
| Recall | 0.6971 |
| Accuracy | 0.9599 |
| F1 Score | 0.7605 |
| False Positive Rate | 0.0137 |

## What "Drift" Looks Like
- A **meaningful drop in precision** (e.g., below 0.75) would mean the model is producing more false positives — recommending candidates who don't actually qualify.
- A **rise in false positive rate** above 0.05 would be a red flag.
- A **drop in recall** below 0.60 would mean the model is missing too many genuine matches.
- Compare new runs against these baselines in `experiment_log.csv`.

## Experiment Log (`metrics/experiment_log.csv`)
- Each row records a training/evaluation run: `run_id`, `timestamp`, `model`, `precision`, `recall`, `accuracy`, `f1_score`, `false_positive_rate`.
- **Keep appending** to this file with every re-evaluation so there's a full history of model performance over time.
- Reproducibility: same data + seed 42 = same numbers.

## What Is OUT of Scope
- **Payment/gateway logic** — that is a separate workstream and not covered by this pipeline.
- **Advanced ML models** — this is a baseline only. Any model upgrades are a separate project.
- **Real-time data ingestion** — the pipeline runs on static CSV files. Streaming or real-time integration is not implemented.
- **User authentication on the API** — the `/match` endpoint is open. Auth is a separate concern.
