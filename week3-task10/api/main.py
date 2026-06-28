"""
PlaceMux Quality Sign-Off - FastAPI Serving Layer
===================================================
Live-demoable endpoints:
  GET /health
  GET /match/{student_id}/{job_id}
  GET /signoff/report
  GET /signoff/reconciliation
"""

import json
import os
import sys

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.baseline import compute_baseline_score
from src.features import build_features, FEATURE_COLS
from src.explainability import explain_prediction
from src.reconciliation import reconcile_payments
from src.labeling import compute_label

app = FastAPI(
    title="PlaceMux Quality Sign-Off API",
    description="Live demo endpoints for the matching/recommendation regression sign-off.",
    version="1.0.0",
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "models")
REPORT_DIR = os.path.join(PROJECT_ROOT, "reports")

_cache = {}

def _load():
    if "students" not in _cache:
        _cache["students"] = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
        _cache["jobs"] = pd.read_csv(os.path.join(DATA_DIR, "jobs.csv"))
        _cache["events"] = pd.read_csv(os.path.join(DATA_DIR, "monetization_events.csv"))
        _cache["stu_map"] = _cache["students"].set_index("student_id")
        _cache["job_map"] = _cache["jobs"].set_index("job_id")
        _cache["model"] = joblib.load(os.path.join(MODEL_DIR, "match_model.joblib"))
    return _cache

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the PlaceMux Quality Sign-Off API",
        "endpoints": [
            "/health",
            "/match/{student_id}/{job_id}",
            "/signoff/report",
            "/signoff/reconciliation"
        ]
    }

@app.get("/health")
def health():
    c = _load()
    return {
        "status": "ok",
        "students_loaded": len(c["students"]),
        "jobs_loaded": len(c["jobs"]),
        "events_loaded": len(c["events"]),
        "model_loaded": c["model"] is not None,
    }

@app.get("/match/{student_id}/{job_id}")
def match(student_id: str, job_id: str):
    c = _load()
    stu_map, job_map = c["stu_map"], c["job_map"]

    if student_id not in stu_map.index:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
    if job_id not in job_map.index:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    student = stu_map.loc[student_id]
    job = job_map.loc[job_id]

    events = c["events"]
    pair_events = events[(events["student_id"] == student_id) & (events["job_id"] == job_id)]
    if len(pair_events) > 0:
        event = pair_events.iloc[0]
    else:
        event = pd.Series({
            "payment_status": "success",
            "gateway_amount": 0.0,
            "recorded_amount": 0.0,
        })

    baseline = compute_baseline_score(student, job)
    label = compute_label(student, job)
    explanation = explain_prediction(student, job, event, c["model"])

    return {
        "student_id": student_id,
        "job_id": job_id,
        "ground_truth_label": label,
        "baseline": {
            "overlap_count": baseline["overlap_count"],
            "total_required": baseline["total_required"],
            "overlap_ratio": baseline["overlap_ratio"],
            "is_match": baseline["is_match"],
        },
        "model": {
            "prediction": explanation["prediction"],
            "confidence": explanation["confidence"],
        },
        "explanation": explanation["explanation"],
        "skill_breakdown": explanation["skill_breakdown"],
        "top_feature_drivers": explanation["top_feature_drivers"],
    }

@app.get("/signoff/report")
def signoff_report():
    report_path = os.path.join(REPORT_DIR, "evaluation_results.json")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=503, detail="Evaluation not yet run.")
    with open(report_path) as f:
        return json.load(f)

@app.get("/signoff/reconciliation")
def signoff_reconciliation():
    c = _load()
    return reconcile_payments(c["events"])
