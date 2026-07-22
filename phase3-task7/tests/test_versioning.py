import pandas as pd
from src.cold_start_recommender import ColdStartRecommender, MODEL_VERSION
from src.fallback import get_recommendations

jobs = pd.read_json("data/jobs.json")
recommender = ColdStartRecommender()

def test_model_version_constant():
    assert MODEL_VERSION == "1.0.0"
    assert recommender.version == MODEL_VERSION

def test_recommendation_contains_version():
    res = get_recommendations(recommender, ["python"], jobs, k=5, model_available=True)
    assert "model_version" in res
    assert res["model_version"] == MODEL_VERSION

def test_fallback_contains_version():
    res = get_recommendations(recommender, ["python"], jobs, k=5, model_available=False)
    assert "model_version" in res
    assert res["model_version"] == MODEL_VERSION
