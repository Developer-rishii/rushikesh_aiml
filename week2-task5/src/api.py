from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(__file__))
from baseline_matcher import BaselineMatcher
from ranker import CandidateRanker
import joblib

app = FastAPI(title="PlaceMux Match API", version="1.0")

# Load data into memory
try:
    JOBS_DF = pd.read_csv("data/jobs.csv")
    STUDENTS_DF = pd.read_csv("data/students.csv")
    
    matcher = BaselineMatcher(threshold=70.0)
    ranker = CandidateRanker(matcher)
    
    # Optional: Load ML model
    ml_model = joblib.load("models/logistic_regression.pkl")
except Exception as e:
    print(f"Startup warning: {e}")

class MatchRequest(BaseModel):
    job_id: str
    student_id: str

class RankRequest(BaseModel):
    job_id: str

@app.get("/")
def read_root():
    return {"message": "Welcome to PlaceMux Matching API v1"}

@app.post("/match")
def match_candidate(req: MatchRequest):
    job = JOBS_DF[JOBS_DF['job_id'] == req.job_id]
    student = STUDENTS_DF[STUDENTS_DF['student_id'] == req.student_id]
    
    if job.empty:
        raise HTTPException(status_code=404, detail="Job not found")
    if student.empty:
        raise HTTPException(status_code=404, detail="Student not found")
        
    job_series = job.iloc[0]
    student_series = student.iloc[0]
    
    res = matcher.match(job_series, student_series)
    
    if "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
        
    # Also run the ML model prediction to satisfy requirements
    from ml_model import feature_engineering
    temp_df = pd.DataFrame([{
        "job_id": job_series.get('job_id'),
        "student_id": student_series.get('student_id'),
        "required_skills": job_series.get('required_skills', ''),
        "verified_skills": student_series.get('verified_skills', ''),
        "skill_scores": student_series.get('skill_scores', '{}'),
        "experience": student_series.get('experience', 0),
        "experience_required": job_series.get('experience_required', 0)
    }])
    X_feats, _ = feature_engineering(temp_df)
    features = X_feats.values.tolist()
    ml_pred = ml_model.predict(features)[0]
    
    return {
        "job_id": req.job_id,
        "student_id": req.student_id,
        "match_score": round(res["match_score"], 2),
        "match_vector": res["match_vector"],
        "threshold_passed": res["status"] == "eligible",
        "reason": res["reasons_list"],
        "ml_prediction": "Matched" if ml_pred == 1 else "Not Matched"
    }

@app.post("/rank-candidates")
def rank_candidates(req: RankRequest):
    job = JOBS_DF[JOBS_DF['job_id'] == req.job_id]
    if job.empty:
        raise HTTPException(status_code=404, detail="Job not found")
        
    job_series = job.iloc[0]
    
    ranked_list = ranker.rank_candidates(job_series, STUDENTS_DF)
    if not isinstance(ranked_list, list):
        raise HTTPException(status_code=400, detail="Error processing candidates")
        
    # Format output
    output = []
    for i, res in enumerate(ranked_list):
        output.append({
            "rank": i + 1,
            "student_id": res["student_id"],
            "match_score": round(res["match_score"], 2),
            "average_verified_skill_score": round(res.get("average_verified_skill_score", 0), 2),
            "experience_gap": res.get("experience_gap", 0),
            "status": res["status"]
        })
        
    return {"job_id": req.job_id, "shortlist": output}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
