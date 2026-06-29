from fastapi import FastAPI, HTTPException
import pandas as pd
import json
import os
from src.data_loader import load_and_validate_data
from src.model import load_model_and_imputer, predict_session

app = FastAPI(title="PlaceMux Proctoring API")

DATA_PATH = "d:/Placemux-aiml/week4-task11/data/integrity_data_week1.csv"
MODEL_DIR = "d:/Placemux-aiml/week4-task11/src/models"

# Global variables
df = None
clf = None
imputer = None

@app.on_event("startup")
def startup_event():
    global df, clf, imputer
    try:
        df = load_and_validate_data(DATA_PATH)
        clf, imputer = load_model_and_imputer(MODEL_DIR)
    except Exception as e:
        print(f"Warning during startup: {e}")

@app.get("/")
def read_root():
    return {"message": "PlaceMux Proctoring API. Available endpoints: /proctor/check/{session_id}, /hardening/report, /hardening/edge-cases"}

@app.get("/proctor/check/{session_id}")
def check_session(session_id: str):
    if df is None or clf is None:
        raise HTTPException(status_code=500, detail="Data or Model not loaded properly.")
    
    session_data = df[df['session_id'] == session_id]
    if session_data.empty:
        raise HTTPException(status_code=404, detail="Session not found.")
        
    row = session_data.iloc[0]
    result = predict_session(row, clf, imputer)
    
    return result

@app.get("/hardening/report")
def get_report():
    log_path = os.path.join(MODEL_DIR, "experiment_log.json")
    if not os.path.exists(log_path):
        raise HTTPException(status_code=404, detail="Report not found (experiment not run).")
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        if not lines:
            raise HTTPException(status_code=404, detail="Empty log.")
        latest_log = json.loads(lines[-1])
        
    # Format the report to match expectations
    fpr_delta = latest_log['baseline_metrics']['fpr'] - latest_log['model_metrics']['fpr']
    
    return {
        "baseline_vs_model": {
            "baseline": latest_log['baseline_metrics'],
            "model": latest_log['model_metrics']
        },
        "fpr_delta": fpr_delta,
        "segment_breakdown": "Evaluated on held-out test split (see report artifact for more granular breakdown)",
    }

@app.get("/hardening/edge-cases")
def get_edge_cases():
    if df is None:
         raise HTTPException(status_code=500, detail="Data not loaded properly.")
    
    from src.data_loader import is_sensor_fault
    
    # 1. Sensor Fault
    sensor_fault_rows = df[df.apply(is_sensor_fault, axis=1)]
    
    # 2. Duplicates (read raw)
    try:
        raw_df = pd.read_csv(DATA_PATH)
        dups = raw_df[raw_df.duplicated(subset=['session_id'], keep=False)]
    except Exception:
        dups = pd.DataFrame()
        
    # 3. Borderline rows
    borderline_rows = df[(df['tab_switch_count'] == 1) & (df['face_count_anomalies'] == 0) & (df['copy_paste_events'] == 0)]
    
    result = {
        "sensor_fault": {},
        "duplicates": {},
        "borderline": {}
    }
    
    if not sensor_fault_rows.empty:
        sf_row = sensor_fault_rows.iloc[0]
        sf_pred = predict_session(sf_row, clf, imputer)
        result["sensor_fault"] = {
            "session_id": sf_row['session_id'],
            "handling": sf_pred['verdict'],
            "explanation": sf_pred['explanation']
        }
        
    if not dups.empty:
        result["duplicates"] = {
            "raw_count": len(dups),
            "loaded_count": len(df[df['session_id'] == dups.iloc[0]['session_id']]),
            "handling": "Deterministically deduplicated during load_and_validate_data (keep first)"
        }
        
    if not borderline_rows.empty:
        bl_row = borderline_rows.iloc[0]
        bl_pred = predict_session(bl_row, clf, imputer)
        result["borderline"] = {
            "session_id": bl_row['session_id'],
            "confidence": bl_pred['confidence'],
            "verdict": bl_pred['verdict'],
            "explanation": bl_pred['explanation']
        }
        
    return result
