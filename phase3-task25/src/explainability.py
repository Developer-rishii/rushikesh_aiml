"""
Stage B.4/C.4/D.4 - "Make it explainable, safe & demoable: one worked
example - this input, this output, this plain-English reason".
Uses LightGBM's native feature contribution (pred_contrib) so the
explanation is the model's actual decomposition, not a post-hoc guess.
"""
import os, json, pickle
import pandas as pd
from common import get_X, FEATURES

ROOT = os.path.dirname(os.path.dirname(__file__))

def main():
    df = pd.read_csv(f"{ROOT}/data/logs.csv")
    test = df[df.day >= 20].copy()
    with open(f"{ROOT}/registry/models/ranker_v2.0.pkl", "rb") as f:
        model = pickle.load(f)

    qid = test.query_id.value_counts().index[0]
    g = test[test.query_id == qid].reset_index(drop=True)
    X = get_X(g)
    scores = model.predict(X)
    contrib = model.predict(X, pred_contrib=True)  # shape (n, n_features+1), last col = base value

    best_i = scores.argmax()
    row = g.iloc[best_i]
    contribs = dict(zip(FEATURES, contrib[best_i][:-1]))
    base_value = contrib[best_i][-1]
    top_reason = max(contribs, key=lambda k: abs(contribs[k]))

    plain_english = {
        "fit_gap": "the candidate's inferred skill level closely matches this job's seniority",
        "skill_score": "the candidate's overall skill score",
        "exp_years": "the candidate's years of experience",
        "job_seniority": "this job's seniority level",
        "job_comp_level": "this job's compensation band",
    }[top_reason]

    worked_example = {
        "candidate_id": int(row.candidate_id), "job_id": int(row.job_id),
        "input_features": {k: float(row[k]) if k != "fit_gap" else None for k in FEATURES},
        "model_score": float(scores[best_i]),
        "rank_among_shown_jobs": 1,
        "n_jobs_considered": len(g),
        "base_value": float(base_value),
        "feature_contributions": {k: round(float(v), 4) for k, v in contribs.items()},
        "top_driver": top_reason,
        "plain_english_reason": f"This job was ranked #1 for this candidate mainly because "
                                 f"of {plain_english} (contribution {contribs[top_reason]:+.3f} "
                                 f"to the score, vs. a base value of {base_value:.3f}).",
        "what_if_model_unavailable": "the baseline popularity ranker would have shown this "
                                      "candidate the most-clicked job overall instead - see dr_failover_test.json"
    }
    json.dump(worked_example, open(f"{ROOT}/reports/worked_example.json", "w"), indent=2)
    print(json.dumps(worked_example, indent=2))

if __name__ == "__main__":
    main()
