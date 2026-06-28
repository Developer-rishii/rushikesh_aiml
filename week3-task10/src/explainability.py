"""
PlaceMux Quality Sign-Off - Explainability Layer
==================================================
Per-prediction plain-English explanations derived from feature importances
and the actual feature values for a given (student, job) pair.
"""

import numpy as np
import pandas as pd

from src.features import build_features, FEATURE_COLS, _parse_required_levels

SKILLS = ["Python", "JavaScript", "React", "Node.js", "SQL", "Docker", "AWS", "Machine Learning"]


def explain_prediction(student: pd.Series, job: pd.Series, event: pd.Series,
                       model, feature_names: list[str] = FEATURE_COLS) -> dict:
    feats = build_features(student, job, event)
    X = np.array([[feats[c] for c in feature_names]])

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    confidence = round(float(proba[pred]), 4)

    importances = dict(zip(feature_names, model.feature_importances_))
    top_drivers = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

    req_levels = _parse_required_levels(job)
    req_skills = list(req_levels.keys())
    total_req = len(req_skills)

    matched_skills = []
    missing_skills = []
    weak_skills = []
    for sk in req_skills:
        req_lv = req_levels[sk]
        stu_lv = student.get(sk, np.nan)
        if pd.notna(stu_lv) and stu_lv > 0:
            stu_lv = int(stu_lv)
            if stu_lv >= req_lv:
                matched_skills.append(f"{sk} at level {stu_lv} (required {req_lv}) [OK]")
            else:
                weak_skills.append(f"{sk} at level {stu_lv} vs required {req_lv}")
        else:
            missing_skills.append(sk)

    overlap_str = f"Matched on {feats['skill_overlap_count']}/{total_req} required skills"
    parts = [overlap_str]

    if matched_skills:
        parts.append("Strong: " + "; ".join(matched_skills[:3]))
    if weak_skills:
        parts.append("Weak: " + "; ".join(weak_skills))
    if missing_skills:
        parts.append("Missing: " + ", ".join(missing_skills))

    parts.append(
        f"Model confidence: {confidence:.2f}, driven mainly by "
        + ", ".join(d[0].replace("_", " ") for d in top_drivers[:3])
    )

    verdict = "GOOD MATCH" if pred == 1 else "NOT A MATCH"
    parts.insert(0, f"[{verdict}]")

    return {
        "prediction": pred,
        "confidence": confidence,
        "explanation": ". ".join(parts) + ".",
        "skill_breakdown": {
            "matched": matched_skills,
            "weak": weak_skills,
            "missing": missing_skills,
        },
        "top_feature_drivers": [{"feature": d[0], "importance": round(d[1], 4)} for d in top_drivers],
        "features": feats,
    }
