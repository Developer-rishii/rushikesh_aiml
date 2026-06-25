from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import pandas as pd
import os
import sys
import uvicorn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from matcher import JobMatcher

app = FastAPI(title="Matching Tune API", version="1.0.0")

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

class CandidateProfile(BaseModel):
    Candidate_ID: str
    Skills: str = ""
    Experience_Years: float = 0.0
    Education: str = ""
    Certifications: str = ""
    Projects: float = 0.0

# Initialize Matcher
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/logistic_regression.joblib')
DATA_PATH = os.path.join(BASE_DIR, '../data/jobs.csv')

matcher = JobMatcher(model_path=MODEL_PATH)

# Load jobs pool
try:
    jobs_df = pd.read_csv(DATA_PATH)
    jobs_pool = jobs_df.to_dict('records')
except Exception as e:
    jobs_pool = []
    print(f"Warning: Could not load jobs pool. {e}")

@app.post("/match")
def match_candidate(candidate: CandidateProfile):
    if not jobs_pool:
        raise HTTPException(status_code=500, detail="Jobs pool is empty or not loaded.")
        
    cand_dict = {
        'Candidate ID': candidate.Candidate_ID,
        'Skills': candidate.Skills,
        'Experience Years': candidate.Experience_Years,
        'Education': candidate.Education,
        'Certifications': candidate.Certifications,
        'Projects': candidate.Projects
    }
    
    try:
        ranked_results = matcher.rank_jobs(cand_dict, jobs_pool)
        
        # Return top 10 matches for the API response to avoid huge payloads
        top_10 = ranked_results[:10]
        
        response = {
            "ranked_jobs": [res['job_id'] for res in top_10],
            "match_scores": [res['final_score'] for res in top_10],
            "explanations": [res['explanation'] for res in top_10]
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
