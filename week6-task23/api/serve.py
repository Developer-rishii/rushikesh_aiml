"""
api/serve.py

FastAPI inference layer utilizing Feature Store and Model Registry. Contract:

POST /match
{
  "student_id": "S...",
  "job_id": "J..."
}
->
{
  "score": 0.73, "decision": 1, "model_version": "v3",
  "explanation": "Match score for ... -> RECOMMENDED. ..."
}
"""
import sys
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import FEATURE_NAMES
from src.model import MatchModel
from src.explain import explain_match
from src.feature_store import FeatureStore
from src.registry import ModelRegistry

app = FastAPI(title="PlaceMux MLOps Match Scoring API")

fs = FeatureStore(os.path.join(os.path.dirname(__file__), "..", "data"))
registry = ModelRegistry()

_model: Optional[MatchModel] = None
_model_version: Optional[str] = None


def get_model() -> MatchModel:
    global _model, _model_version
    try:
        prod_path = registry.get_production_model_path()
        if _model is None or prod_path != getattr(_model, "_registry_path", None):
            _model = MatchModel.load(prod_path)
            _model._registry_path = prod_path
    except Exception as e:
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "models", "model_latest.joblib")):
            raise HTTPException(status_code=503, detail=f"Model not trained/loaded yet: {e}")
        prod_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_latest.joblib")
        if _model is None or prod_path != getattr(_model, "_registry_path", None):
            _model = MatchModel.load(prod_path)
            _model._registry_path = prod_path
    return _model


class MatchRequest(BaseModel):
    student_id: str
    job_id: str


@app.post("/match")
def score_match(req: MatchRequest):
    model = get_model()
    try:
        feat = fs.get_online_features(req.student_id, req.job_id)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not build features: {e}")

    feat_df = pd.DataFrame([{**feat, "good_match": 0}])
    score = float(model.predict_proba(feat_df)[0])
    decision = int(score >= model.threshold)
    
    # We don't need full dicts for explain_match if we rewrite the explanation slightly,
    # but to preserve the old explain logic, we can fetch the skills directly from the DataFrames
    student_skills = eval(fs._s_idx.loc[req.student_id]["skills_json"]) if req.student_id in fs._s_idx.index else {}
    job_skills = eval(fs._j_idx.loc[req.job_id]["required_skills_json"]) if req.job_id in fs._j_idx.index else {}
    
    explanation = explain_match(req.student_id, req.job_id, student_skills,
                                 job_skills, feat, score, decision, model)
    return {"score": round(score, 4), "decision": decision,
            "model_version": model.version, "explanation": explanation}


@app.get("/health")
def health():
    try:
        prod_path = registry.get_production_model_path()
        exists = os.path.exists(prod_path)
    except:
        prod_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_latest.joblib")
        exists = os.path.exists(prod_path)
    return {"status": "ok" if exists else "model_missing", "model_path": prod_path}
