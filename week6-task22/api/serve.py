"""
api/serve.py

FastAPI inference layer. Contract:

POST /match
{
  "student_id": "S...",
  "job_id": "J...",
  "student_skills": {"python": 0.8, ...},
  "job_skills": {"python": 0.9, ...},
  "years_gap": 1
}
->
{
  "score": 0.73, "decision": 1, "model_version": "v3",
  "explanation": "Match score for ... -> RECOMMENDED. ..."
}

Edge cases handled explicitly (see src/features.py + tests/):
  - empty student_skills / job_skills -> still returns a valid (low) score,
    doesn't crash.
  - malformed payload -> 422 with a clear message, not a 500.
"""
import sys
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import build_features_row, FEATURE_NAMES
from src.model import MatchModel
from src.explain import explain_match

app = FastAPI(title="PlaceMux Match Scoring API")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_latest.joblib")
_model: Optional[MatchModel] = None


def get_model() -> MatchModel:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=503, detail="Model not trained/loaded yet.")
        _model = MatchModel.load(MODEL_PATH)
    return _model


class MatchRequest(BaseModel):
    student_id: str
    job_id: str
    student_skills: Dict[str, float] = Field(default_factory=dict)
    job_skills: Dict[str, float] = Field(default_factory=dict)
    years_gap: int = 0


@app.post("/match")
def score_match(req: MatchRequest):
    model = get_model()
    try:
        feat = build_features_row(req.student_skills, req.job_skills, req.years_gap)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not build features: {e}")

    feat_df = pd.DataFrame([{**feat, "good_match": 0}])
    score = float(model.predict_proba(feat_df)[0])
    decision = int(score >= model.threshold)
    explanation = explain_match(req.student_id, req.job_id, req.student_skills,
                                 req.job_skills, feat, score, decision, model)
    return {"score": round(score, 4), "decision": decision,
            "model_version": model.version, "explanation": explanation}


@app.get("/health")
def health():
    exists = os.path.exists(MODEL_PATH)
    return {"status": "ok" if exists else "model_missing", "model_path": MODEL_PATH}
