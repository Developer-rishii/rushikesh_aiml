import os
import sys
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

# Allow imports to work both via `uvicorn src.api:app` and direct `from api import ...`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from feature_engineering import extract_features_for_pair
from explainability import generate_explanation
from baseline_matcher import calculate_baseline_score

app = FastAPI(title="PlaceMux Quality Baseline API")

model = None
scaler = None

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "..", "data")
model_path = os.path.join(current_dir, "..", "models", "baseline_model.pkl")

try:
    candidates_df, jobs_df = load_data(data_dir)
except Exception as e:
    candidates_df, jobs_df = None, None

@app.on_event("startup")
async def startup_event():
    global model, scaler
    try:
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
            model = artifact["model"]
            scaler = artifact["scaler"]
    except Exception as e:
        print(f"Warning: Model could not be loaded: {e}")

class MatchRequest(BaseModel):
    candidate_id: int
    job_id: int

@app.post("/match")
async def match_endpoint(req: MatchRequest):
    if candidates_df is None or jobs_df is None:
        raise HTTPException(status_code=500, detail="Data files candidate_profiles.csv or jobs.csv missing")
        
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Corrupted/unreadable model file baseline_model.pkl")
        
    candidate_rows = candidates_df[candidates_df['candidate_id'] == req.candidate_id]
    if len(candidate_rows) == 0:
        return {"error": "Candidate not found"}
        
    job_rows = jobs_df[jobs_df['job_id'] == req.job_id]
    if len(job_rows) == 0:
        return {"error": "Job not found"}
        
    c_dict = candidate_rows.iloc[0].to_dict()
    j_dict = job_rows.iloc[0].to_dict()
    
    if pd.isna(c_dict.get('skills')) or str(c_dict.get('skills')).strip() == '' or str(c_dict.get('skills')) == 'nan':
        return {"error": "Candidate skills missing"}
        
    features_dict = extract_features_for_pair(c_dict, j_dict)
    baseline_score = features_dict['required_skill_coverage']
    
    X_pred = pd.DataFrame([features_dict])[['skill_overlap_percentage', 'experience_gap', 'education_match', 'certification_match_count', 'required_skill_coverage']]
    X_scaled = scaler.transform(X_pred)
    
    pred_prob = model.predict_proba(X_scaled)[0][1]
    prediction = int(model.predict(X_scaled)[0])
    
    explanation = generate_explanation(features_dict)
    
    return {
        "candidate_id": req.candidate_id,
        "job_id": req.job_id,
        "baseline_score": baseline_score,
        "prediction_score": round(pred_prob * 100, 2),
        "prediction": prediction,
        "explanation": explanation
    }
