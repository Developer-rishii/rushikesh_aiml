"""
PlaceMux Quality Sign-Off - Feature Engineering
=================================================
Derives ML features from (student, job, event) triples.
No leakage from the label itself - all features come from raw data.
"""

import numpy as np
import pandas as pd

SKILLS = ["Python", "JavaScript", "React", "Node.js", "SQL", "Docker", "AWS", "Machine Learning"]

FEATURE_COLS = [
    "skill_overlap_count",
    "weighted_coverage_score",
    "mean_level_delta",
    "min_level_delta",
    "max_level_delta",
    "num_missing_required",
    "years_of_exposure",
    "price_tier_free",
    "price_tier_basic",
    "price_tier_premium",
    "pay_success",
    "pay_failed",
    "pay_pending",
    "pay_refunded",
]


def _parse_required_levels(row: pd.Series) -> dict[str, int]:
    """Parse 'Python:3|React:4' -> {'Python': 3, 'React': 4}."""
    raw = row.get("required_levels", "")
    if pd.isna(raw) or raw == "":
        return {}
    pairs = {}
    for chunk in str(raw).split("|"):
        if ":" in chunk:
            sk, lv = chunk.rsplit(":", 1)
            try:
                pairs[sk.strip()] = int(lv)
            except ValueError:
                pass
    return pairs


def build_features(student: pd.Series, job: pd.Series,
                   event: pd.Series) -> dict:
    """Build feature dict for a single (student, job, event) triple."""
    req_levels = _parse_required_levels(job)
    req_skills = list(req_levels.keys())
    total_required = len(req_skills)

    overlap_count = 0
    weighted_num = 0.0
    weighted_den = 0.0
    deltas = []
    missing = 0

    for sk in req_skills:
        req_lv = req_levels[sk]
        weighted_den += req_lv
        stu_lv = student.get(sk, np.nan)
        if pd.notna(stu_lv) and stu_lv > 0:
            overlap_count += 1
            weighted_num += min(float(stu_lv), float(req_lv))
            deltas.append(float(stu_lv) - float(req_lv))
        else:
            missing += 1

    coverage = weighted_num / weighted_den if weighted_den > 0 else 0.0
    mean_delta = float(np.mean(deltas)) if deltas else -5.0
    min_delta = float(np.min(deltas)) if deltas else -5.0
    max_delta = float(np.max(deltas)) if deltas else -5.0

    tier = str(job.get("price_tier", "free")).lower()
    status = str(event.get("payment_status", "success")).lower()

    return {
        "skill_overlap_count": overlap_count,
        "weighted_coverage_score": round(coverage, 4),
        "mean_level_delta": round(mean_delta, 4),
        "min_level_delta": round(min_delta, 4),
        "max_level_delta": round(max_delta, 4),
        "num_missing_required": missing,
        "years_of_exposure": float(student.get("years_of_exposure", 0)),
        "price_tier_free": int(tier == "free"),
        "price_tier_basic": int(tier == "basic"),
        "price_tier_premium": int(tier == "premium"),
        "pay_success": int(status == "success"),
        "pay_failed": int(status == "failed"),
        "pay_pending": int(status == "pending"),
        "pay_refunded": int(status == "refunded"),
    }


def build_feature_matrix(students: pd.DataFrame, jobs: pd.DataFrame,
                         events: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix for all events. Returns DataFrame with FEATURE_COLS + IDs."""
    stu_map = students.set_index("student_id")
    job_map = jobs.set_index("job_id")

    rows = []
    for _, ev in events.iterrows():
        sid, jid = ev["student_id"], ev["job_id"]
        if sid not in stu_map.index or jid not in job_map.index:
            continue
        feats = build_features(stu_map.loc[sid], job_map.loc[jid], ev)
        feats["student_id"] = sid
        feats["job_id"] = jid
        feats["application_id"] = ev["application_id"]
        feats["payment_status"] = ev["payment_status"]
        feats["price_tier"] = ev.get("price_tier", job_map.loc[jid].get("price_tier", "free"))
        rows.append(feats)

    return pd.DataFrame(rows)
