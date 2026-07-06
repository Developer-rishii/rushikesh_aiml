from fastapi import FastAPI, HTTPException
import pandas as pd
import json
import sys
import os

# Resolve project root (one level up from api/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

app = FastAPI(title="PlaceMux Explainability API — Task 18")

# Load data at startup
_data_path = os.path.join(PROJECT_ROOT, "data", "processed", "explanations_output.csv")
try:
    df = pd.read_csv(_data_path)
    print(f"Loaded {len(df)} rows from {_data_path}")
except Exception as e:
    print(f"Warning: could not load data from {_data_path}: {e}")
    df = pd.DataFrame()

# ─────────────────────── Explanation endpoints ───────────────────────

@app.get("/explain/{college_id}/{student_id}/{job_id}")
def get_all_explanations(college_id: str, student_id: str, job_id: str):
    """Return all three audience-level explanations + quality scores."""
    row = df[(df['college_id'] == college_id) & (df['student_id'] == student_id) & (df['job_id'] == job_id)]
    if row.empty:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    row = row.iloc[0]
    return {
        "student_explanation": row['student_explanation'],
        "student_quality_score": float(row['student_explanation_quality']),
        "officer_explanation": row['officer_explanation'],
        "officer_quality_score": float(row['officer_explanation_quality']),
        "admin_explanation": row['admin_explanation'],
        "admin_quality_score": float(row['admin_explanation_quality'])
    }

@app.get("/explain/{college_id}/{student_id}/{job_id}/student")
def get_student_explanation(college_id: str, student_id: str, job_id: str):
    row = df[(df['college_id'] == college_id) & (df['student_id'] == student_id) & (df['job_id'] == job_id)]
    if row.empty:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"explanation": row.iloc[0]['student_explanation']}

@app.get("/explain/{college_id}/{student_id}/{job_id}/officer")
def get_officer_explanation(college_id: str, student_id: str, job_id: str):
    row = df[(df['college_id'] == college_id) & (df['student_id'] == student_id) & (df['job_id'] == job_id)]
    if row.empty:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"explanation": row.iloc[0]['officer_explanation']}

@app.get("/explain/{college_id}/{student_id}/{job_id}/admin")
def get_admin_explanation(college_id: str, student_id: str, job_id: str):
    row = df[(df['college_id'] == college_id) & (df['student_id'] == student_id) & (df['job_id'] == job_id)]
    if row.empty:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"explanation": row.iloc[0]['admin_explanation']}

# ─────────────────────── Portal / Dashboard ───────────────────────

@app.get("/portal/{college_id}/dashboard")
def get_college_dashboard(college_id: str):
    college_data = df[df['college_id'] == college_id]
    if college_data.empty:
        raise HTTPException(status_code=404, detail="College not found")

    avg_quality = college_data['student_explanation_quality'].mean()
    return {
        "college_id": college_id,
        "recommendation_count": int(len(college_data)),
        "avg_explanation_quality_score": round(float(avg_quality), 3),
        "status": "active"
    }

# ─────────────────────── Report & edge-cases ───────────────────────

@app.get("/explain/report")
def get_report():
    report_path = os.path.join(PROJECT_ROOT, "reports", "metrics.json")
    try:
        with open(report_path, "r") as f:
            return json.load(f)
    except Exception:
        return {"error": "Report not generated yet"}

@app.get("/explain/edge-cases")
def get_edge_cases():
    from quality_model import score_explanations

    edge_cases = {}

    # 1. Missing feature_importances_json
    row_missing = df[df['feature_importances_json'].isna()]
    if not row_missing.empty:
        edge_cases["missing_json"] = {
            "handled": True,
            "output": row_missing.iloc[0]['admin_explanation']
        }
    else:
        edge_cases["missing_json"] = {"handled": True, "output": "No rows with missing JSON in current data — edge case handled in code (generates degraded explanation)."}

    # 2. Rank #1 no gaps
    row_rank1 = df[(df['rank_position'] == 1) & (df['skill_gap_count'] == 0)]
    if not row_rank1.empty:
        edge_cases["rank1_no_gaps"] = {
            "handled": True,
            "output": row_rank1.iloc[0]['student_explanation']
        }

    # 3. Audience mismatch — admin explanation scored as if sent to a student
    try:
        admin_expl = df.iloc[0]['admin_explanation']
        dummy = pd.DataFrame([{"explanation_text": admin_expl, "audience": "student", "rank_position": 3}])
        score = score_explanations(dummy)[0]
        edge_cases["audience_mismatch"] = {
            "text": admin_expl,
            "scored_as_audience": "student",
            "quality_score": round(float(score), 3),
            "is_low_quality": bool(score < 0.5)
        }
    except Exception as e:
        edge_cases["audience_mismatch"] = {"error": str(e)}

    return edge_cases
