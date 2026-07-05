"""
api.py

FastAPI inference service for Recommendation v1.
Enforces tenant isolation and provides explainable recommendations.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import pandas as pd
import joblib
import json

from features import FeatureEngineer
from explain import generate_explanation

# Global state
model = None
features_to_use = None
fe = None
jobs_df = None
students_df = None

@asynccontextmanager
async def lifespan(app):
    global model, features_to_use, fe, jobs_df, students_df
    try:
        artifact = joblib.load("model.pkl")
        model = artifact["model"]
        features_to_use = artifact["features_to_use"]
        
        fe = FeatureEngineer()
        fe.load_context()
        fe.priors = artifact["college_priors"]
        
        jobs_df = pd.read_csv("data/jobs.csv")
        students_df = pd.read_csv("data/students.csv")
        print("API Startup: Loaded model and context successfully.")
    except Exception as e:
        print(f"Warning: Failed to load model or context on startup: {e}")
    yield

app = FastAPI(title="Rec v1 API", lifespan=lifespan)

class RecommendRequest(BaseModel):
    student_id: str

class ReverseRecommendRequest(BaseModel):
    job_id: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommend")
def recommend_jobs(req: RecommendRequest, x_college_id: str = Header(...)):
    """Given a student, return top 5 ranked jobs with explanations."""
    student_id = req.student_id
    
    # 1. Tenant Isolation & Existance Check
    student_meta = fe.student_map.get(student_id)
    if not student_meta:
        raise HTTPException(status_code=404, detail="Student not found")
        
    if student_meta["college_id"] != x_college_id:
        raise HTTPException(status_code=403, detail="Cross-tenant access forbidden.")
        
    # 2. Edge Case: Student with zero skills (Cold start)
    if student_id not in fe.student_skills_map or len(fe.student_skills_map[student_id]) == 0:
        return {
            "student_id": student_id,
            "recommendations": [],
            "message": "Cold start: Student has no verified skills. Please update profile."
        }
        
    # 3. Create combinations with all jobs
    job_ids = list(fe.job_map.keys())
    pairs = pd.DataFrame({"student_id": [student_id] * len(job_ids), "job_id": job_ids})
    
    # 4. Feature Extraction & Scoring
    features_df = fe.transform(pairs)
    X = features_df[features_to_use]
    
    scores = model.predict_proba(X)[:, 1]
    features_df["score"] = scores
    features_df["job_id"] = job_ids
    
    # 5. Rank and Explain
    top_jobs = features_df.sort_values(by="score", ascending=False).head(5)
    
    results = []
    for _, row in top_jobs.iterrows():
        # Edge Case: Job with no requirements (caught by overlap_ratio=1.0 logic in features, explained in explain.py)
        explanation = generate_explanation(row.to_dict())
        results.append({
            "job_id": row["job_id"],
            "score": float(row["score"]),
            "explanation": explanation
        })
        
    return {"student_id": student_id, "recommendations": results}

@app.post("/recommend/reverse")
def recommend_students(req: ReverseRecommendRequest, x_college_id: str = Header(...)):
    """Given a job, return top 5 students FROM THIS COLLEGE ONLY."""
    job_id = req.job_id
    
    if job_id not in fe.job_map:
        raise HTTPException(status_code=404, detail="Job not found")
        
    # Filter students strictly by college
    college_students = [sid for sid, meta in fe.student_map.items() if meta["college_id"] == x_college_id]
    
    if not college_students:
        return {"job_id": job_id, "recommendations": [], "message": "No students found in this college."}
        
    pairs = pd.DataFrame({"student_id": college_students, "job_id": [job_id] * len(college_students)})
    
    features_df = fe.transform(pairs)
    X = features_df[features_to_use]
    
    scores = model.predict_proba(X)[:, 1]
    features_df["score"] = scores
    features_df["student_id"] = college_students
    
    top_students = features_df.sort_values(by="score", ascending=False).head(5)
    
    results = []
    for _, row in top_students.iterrows():
        explanation = generate_explanation(row.to_dict())
        results.append({
            "student_id": row["student_id"],
            "score": float(row["score"]),
            "explanation": explanation
        })
        
    return {"job_id": job_id, "recommendations": results}
