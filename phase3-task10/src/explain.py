"""
explain.py
----------
"Produce one worked example: this input, this output, this plain-English
reason" (required for each of deliverables B, C, D by the study guide).

Picks one real held-out query, ranks it with the treatment model, and for
each top-5 candidate explains the ranking in plain English by comparing
that candidate's features to the query's average — no SHAP/black box
dependency needed (none is installed offline; GradientBoostingRegressor's
own feature_importances_ plus a per-candidate delta-from-average is
sufficient and easy to defend to a non-technical stakeholder).
"""

import json

import pandas as pd

from features import FEATURE_COLUMNS, compute_training_features
from train_ranker import train

FEATURE_LABELS = {
    "skill_match": "skill match to the job",
    "experience_years": "years of experience",
    "distance_km": "distance from job location",
    "past_ctr": "past engagement rate",
    "embedding_sim": "profile/job semantic similarity",
    "recency_days": "how recently the job was posted",
}


def explain_query(hist: pd.DataFrame, treatment, qid: int, as_of_day: int, top_k=5):
    group = hist[hist.query_id == qid]
    feats = compute_training_features(group, as_of_day)
    scores = treatment.score(group, as_of_day)
    feats = feats.assign(model_score=scores).sort_values("model_score", ascending=False)

    importances = dict(zip(FEATURE_COLUMNS, treatment.model.feature_importances_))
    signs = treatment.feature_sign
    top_feature = max(importances, key=importances.get)

    examples = []
    query_means = feats[FEATURE_COLUMNS].mean()
    for _, row in feats.head(top_k).iterrows():
        deltas = {c: float(row[c] - query_means[c]) for c in FEATURE_COLUMNS}
        # score each feature by how much it plausibly PUSHED the ranking up:
        # importance x delta-from-average x direction of that feature's
        # real relationship with relevance (e.g. distance is negative)
        contribution = {c: deltas[c] * importances[c] * signs[c] for c in FEATURE_COLUMNS}
        best_reason_feat = max(contribution, key=contribution.get)
        value_direction = "higher" if deltas[best_reason_feat] > 0 else "lower"
        helps_because = (
            "more of this trait raises relevance" if signs[best_reason_feat] > 0
            else "less of this trait raises relevance"
        )
        examples.append(
            {
                "candidate_id": row["candidate_id"],
                "model_score": float(row["model_score"]),
                "plain_english_reason": (
                    f"Ranked highly mainly because of {value_direction}-than-average "
                    f"{FEATURE_LABELS[best_reason_feat]} "
                    f"(this candidate: {row[best_reason_feat]:.2f} vs query average: "
                    f"{query_means[best_reason_feat]:.2f}), and {helps_because} for this feature."
                ),
            }
        )

    return {
        "query_id": int(qid),
        "top_model_feature_overall": FEATURE_LABELS[top_feature],
        "feature_importances": {FEATURE_LABELS[k]: round(float(v), 3) for k, v in importances.items()},
        "top_5_ranked_candidates": examples,
        "model_unavailable_behavior": (
            "If the treatment model is unavailable, this endpoint serves the baseline "
            "heuristic ranking instead (see failure_test.py) rather than failing the request."
        ),
    }


if __name__ == "__main__":
    hist = pd.read_csv("data/historical_logs.csv")
    treatment, _, test_df = train(hist)
    as_of_day = int(hist["posted_day"].max())
    qid = int(test_df["query_id"].iloc[0])

    worked_example = explain_query(hist, treatment, qid, as_of_day)
    with open("artifacts/worked_example.json", "w") as f:
        json.dump(worked_example, f, indent=2)
    print(json.dumps(worked_example, indent=2))
