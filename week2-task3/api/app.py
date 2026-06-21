from fastapi import FastAPI, HTTPException
import sys
import os

# Ensure src module can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ranking import Ranker

app = FastAPI(title="PlaceMux Search & Discovery API")

try:
    ranker = Ranker(model_path="models/model.pkl", students_path="data/students.csv", jobs_path="data/jobs.csv")
except Exception as e:
    print(f"Warning: Could not initialize Ranker. Models or data might be missing. {e}")
    ranker = None

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/recommend-jobs/{student_id}")
def recommend_jobs(student_id: str):
    if not ranker:
        raise HTTPException(status_code=500, detail="Ranker not initialized.")
    results = ranker.rank_jobs_for_student(student_id)
    if isinstance(results, dict) and "error" in results:
        raise HTTPException(status_code=404, detail=results["error"])
    return results

@app.get("/recommend-candidates/{job_id}")
def recommend_candidates(job_id: str):
    if not ranker:
        raise HTTPException(status_code=500, detail="Ranker not initialized.")
    results = ranker.rank_candidates_for_job(job_id)
    if isinstance(results, dict) and "error" in results:
        raise HTTPException(status_code=404, detail=results["error"])
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
