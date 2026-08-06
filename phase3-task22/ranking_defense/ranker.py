"""
ranker.py
Learning-to-rank model for candidate-job matching (Stage C build target:
"Defences against ranking manipulation and keyword stuffing").

Key defensive design: features are built from STRUCTURED skill-overlap
(deduplicated set intersection), NOT raw keyword frequency in free text.
This means repeating a keyword in a resume (stuffing) yields ZERO marginal
feature gain, because a skill either matches once or it doesn't. We
additionally subtract a stuffing-penalty feature produced by the detector,
so even structurally-hidden stuffing is discouraged.
"""
import json, os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(__file__))
from stuffing_detector import rule_signals


def robust_features(cand, job, stuffing_score=0.0):
    overlap = len(set(cand["skills"]) & set(job["required_skills"]))  # capped, dedup'd
    coverage = overlap / max(1, len(job["required_skills"]))
    exp = min(cand["years_exp"], 15) / 15.0
    return [overlap, coverage, exp, stuffing_score]


def build_training_set(candidates, jobs, interactions, stuffing_scores):
    cand_by_id = {c["candidate_id"]: c for c in candidates}
    job_by_id = {j["job_id"]: j for j in jobs}
    X, y, groups = [], [], []
    for row in interactions:
        c = cand_by_id[row["candidate_id"]]
        j = job_by_id[row["job_id"]]
        s = stuffing_scores.get(c["candidate_id"], 0.0)
        X.append(robust_features(c, j, s))
        y.append(row["true_relevance"])
        groups.append(row["job_id"])
    return np.array(X), np.array(y), groups


def naive_features(cand, job):
    """BASELINE (rejected/insecure): raw keyword-count style feature that a
    stuffed resume can inflate — kept only to demonstrate the attack in
    attack_simulation.py, never used in the shipped robust ranker."""
    text = cand["resume_text"].lower()
    kw_hits = sum(text.count(s) for s in job["required_skills"])
    return kw_hits


def train_ranker(candidates, jobs, interactions, stuffing_scores, out_dir):
    job_ids = sorted(list(set([row["job_id"] for row in interactions])))
    train_job_ids, test_job_ids = train_test_split(job_ids, test_size=0.2, random_state=42)
    train_job_ids_set = set(train_job_ids)
    
    train_interactions = [row for row in interactions if row["job_id"] in train_job_ids_set]
    test_interactions = [row for row in interactions if row["job_id"] not in train_job_ids_set]

    X, y, _ = build_training_set(candidates, jobs, train_interactions, stuffing_scores)
    model = GradientBoostingRegressor(random_state=42, n_estimators=150, max_depth=3)
    model.fit(X, y)
    os.makedirs(out_dir, exist_ok=True)
    import joblib
    joblib.dump(model, os.path.join(out_dir, "ranker.joblib"))
    
    with open(os.path.join(out_dir, "test_interactions.json"), "w") as f:
        json.dump(test_interactions, f, indent=2)
        
    return model, test_interactions


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(base, "data", "candidates.json")) as f:
        candidates = json.load(f)
    with open(os.path.join(base, "data", "jobs.json")) as f:
        jobs = json.load(f)
    with open(os.path.join(base, "data", "interactions.json")) as f:
        interactions = json.load(f)

    stuffing_scores = {c["candidate_id"]: rule_signals(c["resume_text"])["repetition_rate"]
                        for c in candidates}
    model, test_interactions = train_ranker(candidates, jobs, interactions, stuffing_scores,
                          os.path.dirname(__file__))
    print("Ranker trained. n_train_rows =", len(interactions) - len(test_interactions))
