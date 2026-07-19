"""
src/api/main.py

Section 5 recommends FastAPI as the serving layer. This is the real
production interface: a recruiter-facing client or another PlaceMux service
calls /predict for a live match score+explanation, and an ops
dashboard/founder calls the /monitor/* endpoints to see whether the model
is still healthy in production - closing the loop Section 1 asks for
("Monitor models live in production").

Run with:
    uvicorn src.api.main:app --reload --port 8000

Note: requires `pip install fastapi uvicorn` (see requirements.txt). The
core monitoring logic this API wraps (src/monitoring/monitor_service.py,
src/inference.py) has zero dependency on FastAPI and is fully covered by
tests/ and scripts/simulate_and_monitor.py, so the monitoring guarantees do
not depend on the web framework being installed/running.
"""

from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import MatchModel
from src.monitoring.monitor_service import MonitoringService

app = FastAPI(title="PlaceMux Match Model - Live Monitoring API", version="1.0.0")

_model: MatchModel = None
_monitor: MonitoringService = None


def get_model() -> MatchModel:
    global _model
    if _model is None:
        _model = MatchModel()
    return _model


def get_monitor() -> MonitoringService:
    global _monitor
    if _monitor is None:
        _monitor = MonitoringService()
    return _monitor


class MatchRequest(BaseModel):
    skill_overlap_score: float = Field(..., ge=0, le=1)
    years_experience: float = Field(..., ge=0)
    experience_gap: float
    resume_parse_confidence: float = Field(..., ge=0, le=1)
    interview_eval_score: float = Field(..., ge=0, le=1)
    communication_score: float = Field(..., ge=0, le=1)
    role_historical_hire_rate: float = Field(..., ge=0, le=1)


@app.get("/health")
def health():
    try:
        get_model()
        return {"status": "ok", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")


@app.post("/predict")
def predict(request: MatchRequest) -> Dict:
    try:
        model = get_model()
        return model.predict_one(request.model_dump(), explain=True)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


@app.post("/monitor/process_batch")
def process_batch(batch: list[MatchRequest]):
    """Manual trigger for demo purposes - normal operation is the scheduled
    loop in scripts/simulate_and_monitor.py / a cron/queue consumer."""
    if not batch:
        raise HTTPException(status_code=400, detail="Empty batch")
    import pandas as pd
    df = pd.DataFrame([b.model_dump() for b in batch])
    df["batch_id"] = -1
    df["is_successful_match"] = -1  # unknown at prediction time
    monitor = get_monitor()
    return monitor.process_batch(df)


@app.get("/monitor/metrics/history")
def metrics_history():
    monitor = get_monitor()
    return monitor.history("batch_metrics").to_dict(orient="records")


@app.get("/monitor/drift/history")
def drift_history():
    monitor = get_monitor()
    return monitor.history("batch_drift").to_dict(orient="records")


@app.get("/monitor/alerts")
def alerts():
    monitor = get_monitor()
    return monitor.history("alerts").to_dict(orient="records")
