from fastapi import FastAPI
import pandas as pd
from src.cold_start_recommender import ColdStartRecommender, MODEL_VERSION
from src.fallback import get_recommendations
from src.explain import explain_recommendation
from src.fairness import audit_fairness

app = FastAPI(title="PlaceMux Cold-Start Recommendation API", version=MODEL_VERSION)
recommender = ColdStartRecommender()
jobs = pd.read_json("data/jobs.json")
MODEL_AVAILABLE = True  # flip to simulate outage for live demo

@app.get("/")
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "model_available": MODEL_AVAILABLE,
        "job_catalog_size": len(jobs)
    }

@app.get("/recommend")
def recommend(user_skills: str = "", k: int = 10):
    skills = [s.strip() for s in user_skills.split(",") if s.strip()] if user_skills else []
    result = get_recommendations(recommender, skills, jobs, k=k, model_available=MODEL_AVAILABLE)

    # Attach metadata & explanations
    if result["source"] == "model" and len(result["job_ids"]) > 0:
        result["explanation"] = explain_recommendation(skills, jobs, result["job_ids"][0], recommender)
        rec_details = recommender.recommend(skills, jobs, k=k, return_details=True)
        result["item_details"] = rec_details.get("details", [])

    result["model_version"] = MODEL_VERSION
    return result

@app.get("/fairness")
def fairness_check():
    users = pd.read_json("data/users.json")
    cold_users = users[users["is_cold_start"]]
    return audit_fairness(cold_users, jobs, recommender)

@app.post("/simulate_outage")
def simulate_outage(down: bool = True):
    global MODEL_AVAILABLE
    MODEL_AVAILABLE = not down
    return {
        "model_available": MODEL_AVAILABLE,
        "status": "OUTAGE_SIMULATED" if down else "OPERATIONAL"
    }