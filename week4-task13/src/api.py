from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.explain import SessionExplainer
from src.baseline import baseline_rule
from fastapi.responses import JSONResponse

app = FastAPI(title="PlaceMux Proctoring FP Reduction API")

class SessionFeature(BaseModel):
    session_id: str
    tab_switch_count: float = Field(..., description="Number of tab switches")
    face_absent_seconds: float = Field(..., description="Total seconds face was not detected")
    multiple_faces_detected: int = Field(..., description="1 if multiple faces detected, else 0")
    audio_anomaly_score: float = Field(..., description="Score from 0.0 to 1.0")
    eye_gaze_offscreen_pct: float = Field(..., description="Percentage of time looking offscreen")
    device_type: str = Field(..., description="desktop, laptop, mobile")
    network_quality: str = Field(..., description="excellent, good, poor")
    session_duration: float = Field(..., description="Duration in seconds")
    time_of_day: str = Field(..., description="morning, afternoon, evening, night")
    candidate_history_flag_rate: float = Field(..., description="Previous flag rate from 0.0 to 1.0")

# Load model explainer at startup
model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.pkl')
try:
    explainer = SessionExplainer(model_path)
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    explainer = None

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={"message": "Malformed or missing features in payload.", "details": exc.errors()},
    )

@app.post("/score-session")
async def score_session(session: SessionFeature):
    if explainer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Fallback to baseline not fully integrated in API yet.")
        
    try:
        # Convert to DataFrame
        df = pd.DataFrame([session.dict()])
        
        # We need to handle adversarial inputs manually if we want to reject them
        # (e.g., negative counts, impossible percentages)
        if session.tab_switch_count < 0 or session.face_absent_seconds < 0:
             return JSONResponse(
                 status_code=400,
                 content={"message": "Malformed data: counts cannot be negative."}
             )
        
        # Get baseline prediction to compute reduction
        # (For a single session, "reduction" doesn't strictly make sense unless we mean 
        # whether the model overturned a baseline FP. Let's provide the baseline label too.)
        base_label = "True Violation" if baseline_rule(df.iloc[0]) == 1 else "False Positive"
        
        # Get prediction and explanation
        label, confidence, reason = explainer.explain(df)
        
        # Calculate overturning
        overturned_baseline = (base_label == "True Violation" and label == "False Positive")
        fp_reduction_vs_baseline = "Yes (Overturned baseline flag)" if overturned_baseline else "No change from baseline"
        
        return {
            "session_id": session.session_id,
            "baseline_label": base_label,
            "label": label,
            "confidence": round(confidence, 4),
            "reason": reason,
            "fp_reduction_vs_baseline": fp_reduction_vs_baseline
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/health")
async def health():
    status = "healthy" if explainer else "degraded (baseline only)"
    return {"status": status}
