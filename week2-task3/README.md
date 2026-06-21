# PlaceMux Search & Discovery

This project implements a complete, demoable AI/ML ranking system that provides:
1. Job Ranking for Students
2. Candidate Ranking for Companies

It follows a simple, explainable approach using Logistic Regression, feature engineering for skill overlaps, and explicitly avoids complex unexplainable models.

## Folder Structure
```
week2-task3/
├── data/                  # Generated dummy data
├── notebooks/             # Exploratory analysis 
├── src/                   # Core Python modules
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── baseline.py
│   ├── train_model.py
│   ├── ranking.py
│   ├── evaluation.py
│   └── explainability.py
├── models/                # Trained ML models
├── experiments/           # ML metrics logs
├── api/                   # FastAPI endpoints
│   └── app.py
├── requirements.txt       # Project dependencies
├── demo.py                # End-to-end demonstration script
└── README.md              # Project documentation
```

## Setup Instructions

1. Ensure Python 3.8+ is installed.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate data and run the end-to-end demonstration:
   ```bash
   python demo.py
   ```

## Training Instructions

If you wish to train the model manually step-by-step:

1. **Generate Data:** `python src/preprocess.py`
2. **Feature Engineering:** `python src/feature_engineering.py`
3. **Run Baseline Model:** `python src/baseline.py`
4. **Train ML Model:** `python src/train_model.py`
5. **Evaluate Model:** `python src/evaluation.py`

## Running API Instructions

To start the FastAPI web server:
```bash
python api/app.py
```
Or via uvicorn directly:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Example API Requests

**1. Recommend Jobs for a Student:**
```bash
curl -X GET "http://localhost:8000/recommend-jobs/S00001"
```

**2. Recommend Candidates for a Job:**
```bash
curl -X GET "http://localhost:8000/recommend-candidates/J00001"
```

**3. Health Check:**
```bash
curl -X GET "http://localhost:8000/health"
```

## Example Outputs

### Ranked Candidate Example
```json
[
  {
    "candidate_id": "S01223",
    "candidate": "Student_1223",
    "score": 92,
    "explanation": [
      "✓ PyTorch matched",
      "✓ Computer Vision matched",
      "✓ Machine Learning matched",
      "✓ Experience requirement satisfied",
      "✓ Education requirement satisfied",
      "✓ Location match satisfied"
    ]
  }
]
```

## Evaluation Metrics Explanation

We measure model performance using standard binary classification metrics:
- **Accuracy:** The percentage of correct matches/non-matches.
- **Precision:** Out of all positive recommendations made, how many were actually successful. High precision means less "noise" in suggestions.
- **Recall:** Out of all actually successful possible matches, how many did the model find.
- **F1 Score:** Harmonic mean of precision and recall.
- **False Positive Rate:** How often the model incorrectly predicts a match when it shouldn't.

Results are appended to `experiments/experiment_log.csv`.
