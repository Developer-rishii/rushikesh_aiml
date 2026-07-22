import pandas as pd
from src.cold_start_recommender import ColdStartRecommender
from src.fallback import get_recommendations, EVERGREEN_JOB_IDS

jobs = pd.read_json("data/jobs.json")
recommender = ColdStartRecommender()

def test_model_unavailable_falls_back_to_popularity():
    out = get_recommendations(recommender, ["python", "sql"], jobs, k=5, model_available=False)
    assert out["source"] == "popularity_fallback"
    assert len(out["job_ids"]) == 5

def test_empty_job_pool_falls_back_to_evergreen():
    empty_jobs = jobs.iloc[0:0]
    out = get_recommendations(recommender, ["python"], empty_jobs, k=5, model_available=True)
    assert out["source"] == "evergreen_fallback"
    assert out["job_ids"] == EVERGREEN_JOB_IDS[:5]

def test_zero_skill_cold_start_never_empty():
    out = get_recommendations(recommender, [], jobs, k=10, model_available=True)
    assert len(out["job_ids"]) == 10  # unknown-skill user still gets a full, non-empty list

def test_recommendation_never_empty_under_all_conditions():
    for model_up in [True, False]:
        for pool in [jobs, jobs.iloc[0:0]]:
            out = get_recommendations(recommender, ["react"], pool, k=5, model_available=model_up)
            assert len(out["job_ids"]) > 0, f"Empty result under model_up={model_up}, pool_size={len(pool)}"