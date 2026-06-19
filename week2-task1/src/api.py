from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

from .feature_engineering import get_feature_spaces
from .matcher import calculate_match
from .explainability import generate_reasons
from .ranking import rank_candidates

app = FastAPI(title="PlaceMux Matching API")

# Global variables for data
STUDENTS_DF = None
JOBS_DF = None

# Paths (assuming api is run from project root)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
STUDENTS_PATH = os.path.join(DATA_DIR, 'students.csv')
JOBS_PATH = os.path.join(DATA_DIR, 'jobs.csv')

@app.on_event("startup")
async def startup_event():
    global STUDENTS_DF, JOBS_DF
    # Load and preprocess data once on startup
    STUDENTS_DF, JOBS_DF = get_feature_spaces(STUDENTS_PATH, JOBS_PATH)

# Models
class MatchRequest(BaseModel):
    student_id: int
    job_id: int

class MatchResponse(BaseModel):
    match_score: int
    status: str
    reasons: List[str]

class CandidateResponse(BaseModel):
    student_id: int
    student_name: str
    match_score: int
    reasons: List[str]

@app.post("/match", response_model=MatchResponse)
async def match_student_job(request: MatchRequest):
    # Find student
    student_row = STUDENTS_DF[STUDENTS_DF['Student ID'] == request.student_id]
    if student_row.empty:
        raise HTTPException(status_code=404, detail=f"Student ID {request.student_id} not found")
    student = student_row.iloc[0].to_dict()
    
    # Find job
    job_row = JOBS_DF[JOBS_DF['Job ID'] == request.job_id]
    if job_row.empty:
        raise HTTPException(status_code=404, detail=f"Job ID {request.job_id} not found")
    job = job_row.iloc[0].to_dict()
    
    # Calculate match
    score, details = calculate_match(student, job)
    reasons = generate_reasons(score, details)
    
    # Determine status (e.g. threshold of 75 for "Matched")
    status = "Matched" if score >= 75 else "Not Matched"
    
    return MatchResponse(
        match_score=score,
        status=status,
        reasons=reasons
    )

@app.get("/top-candidates/{job_id}", response_model=List[CandidateResponse])
async def get_top_candidates(job_id: int, top_n: int = 5):
    # Find job
    job_row = JOBS_DF[JOBS_DF['Job ID'] == job_id]
    if job_row.empty:
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")
    job = job_row.iloc[0].to_dict()
    
    # Rank candidates
    ranked = rank_candidates(job, STUDENTS_DF, top_n=top_n)
    
    return [CandidateResponse(**c) for c in ranked]
