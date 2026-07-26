"""
evaluate.py
===========
Stage B/C/D "Evaluate honestly against a baseline" + Stage E "Show nDCG@k
vs the production baseline on held-out data, with the bias correction
explained."

Splits by job_id (not by row!) so no candidate from a test job ever leaks
into training. Reports offline metrics for:
  heuristic          -- current production baseline
  pairwise_raw       -- pairwise LTR, NO position-bias correction
  pairwise_corrected -- pairwise LTR, IPS position-bias correction (chosen)
  listwise_corrected -- listwise LTR, IPS correction (alternative considered)
against true_relevance (simulator ground truth, held out, never used in
training) -- this is exactly the offline nDCG/MAP-vs-heuristic deliverable.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from features import FEATURE_COLUMNS, build_features
from heuristic_baseline import score as heuristic_score
from train_ltr import prepare_labels, train_pairwise_linear, train_listwise_linear, score_with_weights
from metrics import evaluate_ranking

RNG = np.random.default_rng(123)


def split_by_job(df, train=0.70, val=0.15):
    jobs = df.job_id.unique()
    RNG.shuffle(jobs)
    n = len(jobs)
    train_jobs = set(jobs[: int(n * train)])
    val_jobs = set(jobs[int(n * train): int(n * (train + val))])
    test_jobs = set(jobs[int(n * (train + val)):])
    return (
        df[df.job_id.isin(train_jobs)].copy(),
        df[df.job_id.isin(val_jobs)].copy(),
        df[df.job_id.isin(test_jobs)].copy(),
    )


def run(k=10):
    df = pd.read_csv(ROOT / "data" / "raw_logs.csv")
    train_df, val_df, test_df = split_by_job(df)
    train_df, props, eta = prepare_labels(train_df)

    print(f"Train jobs={train_df.job_id.nunique()}  Val jobs={val_df.job_id.nunique()}  "
          f"Test jobs={test_df.job_id.nunique()}  (held out, never trained on)")
    print(f"Fitted position-bias eta={eta:.3f} from randomized-slice intervention harvesting.\n")

    w_raw = train_pairwise_linear(train_df, "label_raw")
    w_corr = train_pairwise_linear(train_df, "label_corrected")
    w_list = train_listwise_linear(train_df, "label_corrected")

    results = {}
    test_feat = build_features(test_df)

    test_df = test_df.copy()
    test_df["score_heuristic"] = heuristic_score(test_df)
    test_df["score_pairwise_raw"] = score_with_weights(test_feat, w_raw)
    test_df["score_pairwise_corrected"] = score_with_weights(test_feat, w_corr)
    test_df["score_listwise_corrected"] = score_with_weights(test_feat, w_list)

    for name, col in [
        ("heuristic (current production)", "score_heuristic"),
        ("pairwise_raw (no bias correction)", "score_pairwise_raw"),
        ("pairwise_corrected (CHOSEN model)", "score_pairwise_corrected"),
        ("listwise_corrected (alternative considered)", "score_listwise_corrected"),
    ]:
        m = evaluate_ranking(test_df, col, relevance_col="true_relevance", k=k)
        results[name] = m
        print(f"{name:45s} nDCG@{k}={m[f'nDCG@{k}']:.4f}  MAP@{k}={m[f'MAP@{k}']:.4f}  P@5={m['Precision@5']:.4f}")

    lift = (results["pairwise_corrected (CHOSEN model)"][f"nDCG@{k}"]
            / max(results["heuristic (current production)"][f"nDCG@{k}"], 1e-9) - 1) * 100
    print(f"\nnDCG@{k} lift of chosen model vs heuristic: {lift:+.1f}%")

    weights_report = {
        "pairwise_raw": dict(zip(FEATURE_COLUMNS, w_raw.round(4).tolist())),
        "pairwise_corrected": dict(zip(FEATURE_COLUMNS, w_corr.round(4).tolist())),
        "listwise_corrected": dict(zip(FEATURE_COLUMNS, w_list.round(4).tolist())),
        "true_relevance_generating_weights (simulator ground truth, NOT seen by any model)": {
            "skill_match": 0.40, "experience_match": 0.30, "embedding_sim": 0.20,
            "recency": 0.0, "past_response_rate": 0.10, "profile_completeness": 0.0,
        },
    }

    out = {
        "n_train_jobs": int(train_df.job_id.nunique()),
        "n_val_jobs": int(val_df.job_id.nunique()),
        "n_test_jobs": int(test_df.job_id.nunique()),
        "fitted_position_bias_eta": eta,
        "true_position_bias_eta": 1.4,
        "propensities_by_position": props,
        "metrics": results,
        "ndcg_lift_pct_vs_heuristic": lift,
        "learned_weights": weights_report,
    }
    with open(ROOT / "reports" / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    np.save(ROOT / "artifacts" / "w_pairwise_corrected.npy", w_corr)
    np.save(ROOT / "artifacts" / "w_pairwise_raw.npy", w_raw)
    np.save(ROOT / "artifacts" / "w_listwise_corrected.npy", w_list)
    test_df.to_csv(ROOT / "artifacts" / "test_scored.csv", index=False)
    return out


if __name__ == "__main__":
    run()
