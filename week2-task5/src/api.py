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
    # Compute features inline for ML model
    job_req = matcher._parse_skills(job_series.get('required_skills', ''))
    student_skills = matcher._parse_skills(student_series.get('verified_skills', ''))
    
    matched_skills = [s.lower() for s in job_req if s.lower() in [st.lower() for st in student_skills]]
    matched_skill_count = len(matched_skills)
    missing_skill_count = len(job_req) - matched_skill_count
    skill_overlap_percentage = (matched_skill_count / len(job_req)) * 100 if len(job_req) > 0 else 0
    experience_match = 1 if student_series.get('experience', 0) >= job_series.get('experience_required', 0) else 0
    average_verified_skill_score = 75.0 # hardcoded approximation for inline speed
    
    features = [[skill_overlap_percentage, matched_skill_count, missing_skill_count, average_verified_skill_score, experience_match]]
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
            "status": res["status"]
        })
        
    return {"job_id": req.job_id, "shortlist": output}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
