"""
poison_detector.py
Stage D build target: "Detection for poisoned training data". Simulates an
attacker injecting mislabeled interaction rows (high skill-overlap candidate
marked as NOT applied/relevant, or the reverse) into a retraining batch, and
detects them via Isolation Forest on (feature, label) consistency BEFORE
they reach the trainer.
"""
import json, os, random
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ranking_defense"))
from ranker import robust_features


def inject_poison(interactions, candidates, jobs, poison_rate=0.03, seed=42):
    rng = random.Random(seed)
    cand_by_id = {c["candidate_id"]: c for c in candidates}
    job_by_id = {j["job_id"]: j for j in jobs}
    poisoned = [dict(r) for r in interactions]
    n_poison = int(len(poisoned) * poison_rate)
    idxs = rng.sample(range(len(poisoned)), n_poison)
    for i in idxs:
        row = poisoned[i]
        c = cand_by_id[row["candidate_id"]]
        j = job_by_id[row["job_id"]]
        feats = robust_features(c, j)
        # flip the label against what structured features indicate:
        # attacker wants a low-relevance candidate to look highly relevant
        row["true_relevance"] = 0.95 if feats[1] < 0.3 else 0.02
        row["applied"] = True
        row["_is_poison_injected"] = True
    for row in poisoned:
        row.setdefault("_is_poison_injected", False)
    return poisoned, n_poison


def detect(poisoned_interactions, candidates, jobs, out_dir):
    cand_by_id = {c["candidate_id"]: c for c in candidates}
    job_by_id = {j["job_id"]: j for j in jobs}

    X, y_true_poison = [], []
    for row in poisoned_interactions:
        c = cand_by_id[row["candidate_id"]]
        j = job_by_id[row["job_id"]]
        feats = robust_features(c, j)
        X.append(feats + [row["true_relevance"]])
        y_true_poison.append(int(row["_is_poison_injected"]))

    X = np.array(X)
    clf = IsolationForest(contamination=0.05, random_state=42)
    clf.fit(X)
    raw_pred = clf.predict(X)          # -1 = outlier (flagged), 1 = inlier
    y_pred = (raw_pred == -1).astype(int)

    result = {
        "n_rows": len(poisoned_interactions),
        "n_actually_poisoned": int(sum(y_true_poison)),
        "n_flagged_as_outlier": int(y_pred.sum()),
        "precision": round(precision_score(y_true_poison, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true_poison, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true_poison, y_pred, zero_division=0), 4),
        "action_on_flag": "quarantine row -- excluded from next retraining batch, logged for manual audit (never silently retrained on)",
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "poison_eval.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result, y_pred


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(base, "data", "candidates.json")) as f:
        candidates = json.load(f)
    with open(os.path.join(base, "data", "jobs.json")) as f:
        jobs = json.load(f)
    with open(os.path.join(base, "data", "interactions.json")) as f:
        interactions = json.load(f)

    poisoned, n_poison = inject_poison(interactions, candidates, jobs)
    print(f"Injected {n_poison} poisoned rows into {len(interactions)} total")
    result, _ = detect(poisoned, candidates, jobs, os.path.dirname(__file__))
    print(json.dumps(result, indent=2))
