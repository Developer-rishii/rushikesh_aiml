"""
explain.py — "Make it explainable, safe & demoable"
Produces ONE worked example (this input, this output, this plain-English
reason) using the model's own feature contributions, for the live demo.
"""
import json
import os
import pandas as pd
import joblib
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
DEMO = os.path.join(os.path.dirname(__file__), "..", "demo")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from features import build_features, FEATURE_COLUMNS


def main():
    os.makedirs(DEMO, exist_ok=True)
    cands = pd.read_csv(f"{DATA}/candidates.csv")
    jobs = pd.read_csv(f"{DATA}/jobs.csv")
    model = joblib.load(f"{MODELS}/ranker_v1.joblib")

    job = jobs.iloc[3]
    req = set(job["required_skills"].split(","))
    overlaps = cands["skills"].apply(lambda s: len(set(s.split(",")) & req))
    cand = cands.iloc[overlaps.idxmax()]

    imp = pd.DataFrame([{
        "candidate_id": cand["candidate_id"], "job_id": job["job_id"],
        "experience_years": cand["experience_years"],
        "clicked": 0, "shortlisted": 0, "hired": 0,
    }])
    feats = build_features(imp, cands, jobs)
    score = float(model.predict(feats[FEATURE_COLUMNS])[0])

    overlap_skills = sorted(set(cand["skills"].split(",")) & set(job["required_skills"].split(",")))
    missing_skills = sorted(set(job["required_skills"].split(",")) - set(cand["skills"].split(",")))

    reason = (
        f"Ranked because {cand['candidate_id']} matches {len(overlap_skills)}/"
        f"{len(job['required_skills'].split(','))} required skills for "
        f"'{job['tenant_title']}' ({', '.join(overlap_skills) or 'none'}), "
        f"with {cand['experience_years']} years experience. "
        f"Missing: {', '.join(missing_skills) or 'none'}."
    )

    worked_example = {
        "input": {
            "candidate_id": cand["candidate_id"],
            "candidate_skills": cand["skills"],
            "candidate_experience_years": int(cand["experience_years"]),
            "job_id": job["job_id"],
            "job_title_tenant_vocab": job["tenant_title"],
            "job_title_standard": job["std_title"],
            "job_required_skills": job["required_skills"],
        },
        "output": {
            "model_score": round(score, 4),
            "skill_overlap_ratio": round(float(feats["skill_overlap"].iloc[0]), 3),
        },
        "plain_english_reason": reason,
        "fallback_if_model_unavailable": (
            "Candidate would still be shown, ranked by raw skill_overlap "
            "(0-1 fraction of required skills matched) instead of the model "
            "score, with a visible 'baseline ranking' label to the recruiter."
        ),
    }
    with open(f"{DEMO}/worked_example.json", "w") as f:
        json.dump(worked_example, f, indent=2)
    print(json.dumps(worked_example, indent=2))


if __name__ == "__main__":
    main()
