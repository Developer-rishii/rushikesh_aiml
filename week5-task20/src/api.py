"""
FastAPI app for Task 20 — Rec Validation & Go/No-Go.
Exposes validation, drift, dry-run, and go/no-go endpoints.
Also includes college-scoped recommendation endpoints for dry-run testing.
"""
from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import pandas as pd
import json
import os

app = FastAPI(title="PlaceMux Rec Validation & Go/No-Go — Task 20")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# ── State loaded at startup ──────────────────────────────────────────────
MATCHING_DF = None
OUTCOMES_DF = None
STUDENTS_META = None


@app.on_event("startup")
async def startup():
    global MATCHING_DF, OUTCOMES_DF, STUDENTS_META
    try:
        MATCHING_DF = pd.read_csv(os.path.join(DATA_DIR, "fresh_matching.csv"))
        OUTCOMES_DF = pd.read_csv(os.path.join(DATA_DIR, "fresh_outcomes.csv"))
        STUDENTS_META = pd.read_csv(os.path.join(DATA_DIR, "fresh_students_meta.csv"))
        print(f"Loaded {len(MATCHING_DF)} matching rows")
    except Exception as e:
        print(f"Warning: could not load data: {e}")


def _load_json(filename: str):
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{filename} not found. Run the pipeline first.")
    with open(path) as f:
        return json.load(f)


# ── Validation endpoints ─────────────────────────────────────────────────

@app.get("/validation/report")
async def get_validation_report():
    """Full out-of-sample validation metrics."""
    return _load_json("metrics.json")


@app.get("/validation/drift")
async def get_drift_report():
    """Drift-detection result."""
    return _load_json("drift_results.json")


@app.get("/validation/dry-run")
async def get_dry_run():
    """Latest dry-run transcript summary."""
    return _load_json("dry_run_transcript.json")


@app.get("/validation/go-no-go")
async def get_go_no_go():
    """Final Go/No-Go verdict."""
    return _load_json("go_no_go.json")


# ── College-scoped recommendation endpoints (for dry-run testing) ────────

@app.get("/college/{college_id}/recommendations")
async def get_college_recommendations(college_id: str, top_n: int = 5):
    """Get top recommendations for all students in a college."""
    if MATCHING_DF is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    college_data = MATCHING_DF[MATCHING_DF["college_id"] == college_id]
    if college_data.empty:
        raise HTTPException(
            status_code=404,
            detail=f"College '{college_id}' not found or has no data"
        )

    results = []
    for student_id in college_data["student_id"].unique()[:10]:
        sdf = college_data[college_data["student_id"] == student_id]
        top = sdf.nlargest(top_n, "match_score")
        results.append({
            "student_id": student_id,
            "recommendations": top[["job_id", "match_score", "ai_trust_score"]].to_dict("records")
        })
    return {"college_id": college_id, "students": results}


@app.get("/college/{college_id}/student/{student_id}/job/{job_id}")
async def get_student_job_detail(college_id: str, student_id: str, job_id: str):
    """Drill into a specific student-job recommendation with explanation."""
    if MATCHING_DF is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    # Isolation check: student must belong to the requested college
    student_rows = MATCHING_DF[MATCHING_DF["student_id"] == student_id]
    if student_rows.empty:
        raise HTTPException(status_code=404, detail=f"Student '{student_id}' not found")

    actual_college = student_rows.iloc[0]["college_id"]
    if actual_college != college_id:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied: student '{student_id}' belongs to '{actual_college}', not '{college_id}'"
        )

    row = student_rows[student_rows["job_id"] == job_id]
    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No recommendation found for student '{student_id}' and job '{job_id}'"
        )

    r = row.iloc[0]
    return {
        "student_id": student_id,
        "college_id": college_id,
        "job_id": job_id,
        "match_score": float(r["match_score"]),
        "ai_trust_score": float(r["ai_trust_score"]),
        "skill_overlap_count": int(r["skill_overlap_count"]),
        "skill_gap_count": int(r["skill_gap_count"]),
        "explanation": {
            "student_view": f"Your match score for this role is {r['match_score']:.0%}. "
                           f"You have {int(r['skill_overlap_count'])} overlapping skills "
                           f"and {int(r['skill_gap_count'])} gaps to work on.",
            "officer_view": f"Student {student_id} has a {r['match_score']:.0%} match "
                           f"with trust score {r['ai_trust_score']:.2f}. "
                           f"Skill gaps: {int(r['skill_gap_count'])}.",
        },
    }


@app.get("/admin/report")
async def get_admin_report():
    """Aggregated cross-college report."""
    if MATCHING_DF is None:
        raise HTTPException(status_code=500, detail="Data not loaded")

    college_stats = []
    for cid in MATCHING_DF["college_id"].unique():
        cdf = MATCHING_DF[MATCHING_DF["college_id"] == cid]
        college_stats.append({
            "college_id": cid,
            "student_count": int(cdf["student_id"].nunique()),
            "avg_match_score": round(float(cdf["match_score"].mean()), 4),
            "avg_trust_score": round(float(cdf["ai_trust_score"].mean()), 4),
        })
    return {"colleges": college_stats, "total_rows": len(MATCHING_DF)}
