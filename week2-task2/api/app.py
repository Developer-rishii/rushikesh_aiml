from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import os

from src.match_vector import generate_match_vector
from src.threshold_validator import validate_thresholds
from src.scoring import calculate_match_score
from src.explainability import generate_explanation

app = FastAPI(title="Job Matching System API")

class MatchRequest(BaseModel):
    student_id: int
    job_id: int

class MatchResponse(BaseModel):
    match_score: float
    eligible: bool
    match_vector: List[int]
    missing_skills: List[str]
    explanation: str

# In a real app, we'd use a database. Here we load CSVs on startup or lazily.
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    students_path = os.path.join(base_dir, "data", "students.csv")
    jobs_path = os.path.join(base_dir, "data", "jobs.csv")
    
    if not os.path.exists(students_path) or not os.path.exists(jobs_path):
        raise RuntimeError("Data files not found. Please run 'python src/load_data.py' first.")
        
    return pd.read_csv(students_path), pd.read_csv(jobs_path)

@app.post("/match", response_model=MatchResponse)
def match_candidate(request: MatchRequest):
    try:
        students_df, jobs_df = load_data()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    student = students_df[students_df['student_id'] == request.student_id]
    if student.empty:
        raise HTTPException(status_code=404, detail=f"Student ID {request.student_id} not found")
    student_dict = student.iloc[0].to_dict()
    student_skills = {k: v for k, v in student_dict.items() if k != 'student_id'}
        
    job = jobs_df[jobs_df['job_id'] == request.job_id]
    if job.empty:
        raise HTTPException(status_code=404, detail=f"Job ID {request.job_id} not found")
    job_dict = job.iloc[0].to_dict()
    job_reqs = {k: v for k, v in job_dict.items() if k != 'job_id'}
    
    # 1. Match Vector
    match_vector = generate_match_vector(student_skills, job_reqs)
    
    # 2. Validate
    validation = validate_thresholds(student_skills, job_reqs)
    
    # 3. Score
    score = calculate_match_score(match_vector)
    
    # 4. Explain
    explanation = generate_explanation(
        student_skills, job_reqs, score, validation['eligible'], match_vector
    )
    
    return MatchResponse(
        match_score=score,
        eligible=validation['eligible'],
        match_vector=match_vector,
        missing_skills=validation['missing_skills'],
        explanation=explanation
    )
