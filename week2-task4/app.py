from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from src.explainability.explainer import ExplainerEngine
from src.models.baseline import RuleBasedMatcher

app = FastAPI(title="PlaceMux Explainability API", version="1.0.0")

class Student(BaseModel):
    student_id: int
    skills: List[str]
    experience_years: int
    verified_score: int

class Job(BaseModel):
    job_id: int
    required_skills: List[str]
    minimum_score: int

class MatchRequest(BaseModel):
    student: Student
    job: Job
    match_score: Optional[float] = None

class ExplanationResponse(BaseModel):
    match_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    reason: str

explainer = ExplainerEngine()
baseline = RuleBasedMatcher()

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/predict", response_model=ExplanationResponse)
def predict_match(request: MatchRequest):
    try:
        student_dict = request.student.dict()
        job_dict = request.job.dict()
        
        # Convert lists to comma separated strings to match feature engineering
        student_dict["skills"] = ",".join(student_dict["skills"])
        job_dict["required_skills"] = ",".join(job_dict["required_skills"])
        
        # If match score not provided, use baseline
        score = request.match_score
        if score is None:
            baseline_result = baseline.predict(student_dict, job_dict)
            score = baseline_result["match_score"]
            
        explanation = explainer.generate_explanation(student_dict, job_dict, match_score=score)
        
        return ExplanationResponse(
            match_score=explanation["match_score"],
            matched_skills=explanation["matched_skills"],
            missing_skills=explanation["missing_skills"],
            reason=explanation["reason"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
