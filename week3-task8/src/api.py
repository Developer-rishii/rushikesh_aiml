from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
import json
import os
import pickle

from src.baseline_matcher import BaselineMatcher
from src.guardrail import evaluate_guardrail

app = FastAPI(title="PlaceMux Spend-Quality Guardrail API")

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

CANDIDATES_PATH = 'd:/Placemux-aiml/week3-task8/data/candidate_profiles.csv'
JOBS_PATH = 'd:/Placemux-aiml/week3-task8/data/jobs.csv'
METRICS_PATH = 'd:/Placemux-aiml/week3-task8/metrics/guardrail_metrics.json'
MODEL_PATH = 'd:/Placemux-aiml/week3-task8/models/baseline_model.pkl'

def load_threshold():
    if not os.path.exists(METRICS_PATH):
        raise RuntimeError("Threshold config missing. Run threshold calibration first.")
    try:
        with open(METRICS_PATH, 'r') as f:
            data = json.load(f)
            return data['threshold_used']
    except Exception:
        raise RuntimeError("Corrupted threshold config file.")

def check_dependencies():
    if not os.path.exists(CANDIDATES_PATH):
        raise RuntimeError(f"Missing data file: {CANDIDATES_PATH}")
    if not os.path.exists(JOBS_PATH):
        raise RuntimeError(f"Missing data file: {JOBS_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Missing model file: {MODEL_PATH}")
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception:
        raise RuntimeError("Corrupted/unreadable model.")

check_dependencies()
matcher = BaselineMatcher(CANDIDATES_PATH, JOBS_PATH)
threshold = load_threshold()

class GuardrailRequest(BaseModel):
    candidate_id: int
    job_id: int

@app.post("/guardrail-check")
def check_match(request: GuardrailRequest):
    match_data = matcher.match(request.candidate_id, request.job_id)
    
    if match_data is None:
        if not matcher.get_candidate(request.candidate_id):
            return JSONResponse(status_code=404, content={"error": "Candidate not found"})
        if not matcher.get_job(request.job_id):
            return JSONResponse(status_code=404, content={"error": "Job not found"})
        return JSONResponse(status_code=500, content={"error": "Match data unavailable \u2014 cannot evaluate guardrail"})
    
    result = evaluate_guardrail(match_data, threshold)
    
    return {
        "candidate_id": request.candidate_id,
        "job_id": request.job_id,
        "match_score": result["match_score"],
        "fit_status": result["fit_status"],
        "threshold_used": result["threshold"],
        "reason": result["reason"]
    }
