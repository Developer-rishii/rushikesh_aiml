from fastapi import FastAPI, HTTPException
import sys
import os
import pandas as pd
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.explainer import explain_prediction
from src.data_loader import load_and_validate_data
from src.baseline import compute_baseline_metrics
from src.model import train_and_evaluate

app = FastAPI(title="PlaceMux Proctoring Trust Layer")

@app.get("/")
def root():
    return {"message": "Proctoring FP Reduction API"}

@app.get("/proctor/session/{session_id}")
def get_session_decision(session_id: str):
    df = load_and_validate_data()
    session_row = df[df["session_id"] == session_id]
    
    if session_row.empty:
        raise HTTPException(status_code=404, detail="Session not found")
        
    session_data = session_row.iloc[0].to_dict()
    v0_flag = session_data.get("flagged_by_v0")
    
    explanation = explain_prediction(session_data)
    
    return {
        "session_id": session_id,
        "v0_baseline_flag": int(v0_flag) if pd.notna(v0_flag) else None,
        "model_prediction": explanation["prediction"],
        "confidence": explanation["confidence"],
        "verdict": explanation["verdict"],
        "reason": explanation["reason"],
        "fp_pattern": explanation["fp_pattern"]
    }

@app.get("/fp-reduction/report")
def get_fp_report():
    df = load_and_validate_data()
    eval_results = train_and_evaluate(df)
    
    return {
        "metrics": eval_results["metrics"],
        "threshold_used": eval_results["threshold"],
        "absolute_fp_count_before": eval_results["metrics"]["baseline"]["fp_count"],
        "absolute_fp_count_after": eval_results["metrics"]["model"]["fp_count"]
    }

@app.get("/fp-reduction/edge-cases")
def get_edge_cases():
    return {
        "sensor_fault": explain_prediction({
            "tab_switch_count": None, "face_count_anomalies": None, "copy_paste_events": None,
            "network_latency_flag": None, "webcam_dropout_seconds": None
        }),
        "threshold_edge": "Tested via pytest test_threshold_edge_case",
        "duplicate": "Tested via pytest test_duplicate_session_handling",
        "upstream_validation": "Tested via pytest test_upstream_dependency_validation"
    }

@app.get("/fp-reduction/proof")
def get_proof():
    df = load_and_validate_data()
    # Find a known FP that was cleared
    fp_candidates = df[(df["scenario"].isin(["fp_network", "fp_cat", "fp_copypaste"])) & (df["flagged_by_v0"] == 1)]
    fp_session = fp_candidates.iloc[0].to_dict() if not fp_candidates.empty else None
    fp_explanation = explain_prediction(fp_session) if fp_session else None

    # Find a known TP that is still flagged
    tp_candidates = df[(df["scenario"] == "true_violation") & (df["flagged_by_v0"] == 1)]
    tp_session = tp_candidates.iloc[0].to_dict() if not tp_candidates.empty else None
    tp_explanation = explain_prediction(tp_session) if tp_session else None

    return {
        "cleared_fp": {
            "session_id": fp_session["session_id"] if fp_session else None,
            "v0_flag": fp_session["flagged_by_v0"] if fp_session else None,
            "explanation": fp_explanation
        },
        "flagged_tp": {
            "session_id": tp_session["session_id"] if tp_session else None,
            "v0_flag": tp_session["flagged_by_v0"] if tp_session else None,
            "explanation": tp_explanation
        }
    }
