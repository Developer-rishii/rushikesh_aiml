"""
Stage B/C/D step 3: 'Evaluate honestly against a baseline. Evaluate on
held-out data you did not tune on, and report the gap between offline
metric and expected online effect.'

Offline metrics: nDCG@k, MAP@k, Precision@k computed on the held-out test
impressions using the trained model's predicted score to re-rank each
query_id's candidates, evaluated against the (noisy, implicit) click label.

Online proxy: observed CTR by position on the actual log (this is what we
have instead of a live A/B test -- honestly reported as a proxy, not
claimed as a live experiment), split by model_version, which is only
possible BECAUSE Stage C stamped model_version on every impression.

Baseline compared against: a random ranker (shuffle within each job).
"""
import pandas as pd
import numpy as np
import json
import os

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    if len(rels) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(rels) + 2))
    return float(np.sum(rels / discounts))


def ndcg_at_k(rels, k):
    ideal = sorted(rels, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(rels, k) / idcg


def average_precision_at_k(rels, k):
    rels = rels[:k]
    hits, score = 0, 0.0
    for i, r in enumerate(rels, start=1):
        if r > 0:
            hits += 1
            score += hits / i
    return score / hits if hits else 0.0


def evaluate_ranking(df: pd.DataFrame, score_col: str, label_col: str, group_col: str, k=5):
    ndcgs, maps, precisions = [], [], []
    for _, g in df.groupby(group_col):
        g = g.sort_values(score_col, ascending=False)
        rels = g[label_col].tolist()
        ndcgs.append(ndcg_at_k(rels, k))
        maps.append(average_precision_at_k(rels, k))
        precisions.append(np.mean(rels[:k]) if rels else 0.0)
    return {
        f"nDCG@{k}": float(np.mean(ndcgs)),
        f"MAP@{k}": float(np.mean(maps)),
        f"Precision@{k}": float(np.mean(precisions)),
    }


def run():
    test_df = pd.read_csv(os.path.join(ARTIFACTS, "test_predictions.csv"))

    model_metrics = evaluate_ranking(test_df, "pred_score", "label", "job_id")

    rng = np.random.default_rng(0)
    test_df["random_score"] = rng.random(len(test_df))
    baseline_metrics = evaluate_ranking(test_df, "random_score", "label", "job_id")

    events = pd.read_csv(os.path.join(ARTIFACTS, "event_log.csv"))
    impressions = events[events.event_type == "impression"]
    clicks = events[events.event_type == "click"]
    imp_ct = impressions.groupby(["model_version", "position"]).size().rename("impressions")
    click_ct = clicks.groupby(["model_version", "position"]).size().rename("clicks")
    ctr_by_pos_version = pd.concat([imp_ct, click_ct], axis=1).fillna(0)
    ctr_by_pos_version["ctr"] = ctr_by_pos_version["clicks"] / ctr_by_pos_version["impressions"]

    online_by_version = clicks.groupby("model_version").size() / impressions.groupby("model_version").size()

    result = {
        "offline_model_vs_random_baseline": {
            "trained_ranker": model_metrics,
            "random_baseline": baseline_metrics,
            "beats_baseline": model_metrics[f"nDCG@5"] > baseline_metrics[f"nDCG@5"],
        },
        "online_ctr_by_model_version": online_by_version.to_dict(),
        "offline_online_gap_note": (
            "Offline nDCG is computed on noisy implicit click labels (not true "
            "relevance), so it is itself a lower bound. The online CTR-by-version "
            "split is only possible because Stage C stamps model_version on every "
            "impression -- this is the concrete artifact that lets us connect "
            "offline wins to online effect per the Core Concepts section."
        ),
    }
    ctr_by_pos_version.to_csv(os.path.join(ARTIFACTS, "ctr_by_position_and_version.csv"))
    with open(os.path.join(ARTIFACTS, "eval_summary.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    run()
