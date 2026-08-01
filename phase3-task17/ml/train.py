"""
Stage B step 2/3: build on real (logged) data, evaluate honestly against a baseline
on held-out data, log every number for reproducibility.

Approach chosen: pointwise GradientBoostingRegressor trained on the graded relevance
label (0..3), used to RANK candidates per job. Rejected LambdaMART/listwise because
lightgbm/xgboost are unavailable in this offline sandbox (no network to install) -
documented in DESIGN_DECISIONS.md as a deliberate, disclosed substitution, not a
silent downgrade. sklearn's GBR is a defensible pointwise learning-to-rank proxy.
"""
import json
import time
import hashlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
from pathlib import Path

FEATURES = ["skill_overlap", "seniority_gap", "same_location", "recency_days", "candidate_activity"]
REGISTRY_DIR = str(Path(__file__).resolve().parent / "model_registry")


def ndcg_at_k(y_true, y_score, k=10):
    order = np.argsort(-y_score)[:k]
    gains = (2 ** np.array(y_true)[order] - 1)
    discounts = 1 / np.log2(np.arange(2, len(order) + 2))
    dcg = (gains * discounts).sum()
    ideal_order = np.argsort(-np.array(y_true))[:k]
    ideal_gains = (2 ** np.array(y_true)[ideal_order] - 1)
    idcg = (ideal_gains * discounts[:len(ideal_order)]).sum()
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(y_true, y_score, k=5):
    order = np.argsort(-y_score)[:k]
    return float((np.array(y_true)[order] > 0).mean())


def eval_per_job(df, score_col, label_col="label", k_ndcg=10, k_prec=5):
    ndcgs, precs = [], []
    for _, g in df.groupby("job_id"):
        if len(g) < 2:
            continue
        ndcgs.append(ndcg_at_k(g[label_col].values, g[score_col].values, k_ndcg))
        precs.append(precision_at_k(g[label_col].values, g[score_col].values, k_prec))
    return float(np.mean(ndcgs)), float(np.mean(precs))


def main():
    df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "interactions.csv")

    # group split by job_id so no job's impressions leak across train/holdout
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
    train_idx, hold_idx = next(splitter.split(df, groups=df["job_id"]))
    train_df, hold_df = df.iloc[train_idx].copy(), df.iloc[hold_idx].copy()

    model = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.08, random_state=7
    )
    t0 = time.time()
    model.fit(train_df[FEATURES], train_df["label"])
    train_seconds = round(time.time() - t0, 3)

    hold_df["model_score"] = model.predict(hold_df[FEATURES])

    # baseline = popularity/recency heuristic any partner could build without ML
    hold_df["baseline_score"] = (
        -0.5 * hold_df["seniority_gap"] + hold_df["same_location"] - 0.01 * hold_df["recency_days"]
    )

    model_ndcg, model_p5 = eval_per_job(hold_df, "model_score")
    base_ndcg, base_p5 = eval_per_job(hold_df, "baseline_score")

    # feature importances -> used later to build human-readable explanations,
    # NEVER exposed raw to partners (see explain.py contract).
    importances = dict(zip(FEATURES, [round(float(x), 4) for x in model.feature_importances_]))

    import pickle
    import os
    os.makedirs(f"{REGISTRY_DIR}/v1", exist_ok=True)
    with open(f"{REGISTRY_DIR}/v1/model.pkl", "wb") as f:
        pickle.dump(model, f)

    model_hash = hashlib.sha256(open(f"{REGISTRY_DIR}/v1/model.pkl", "rb").read()).hexdigest()[:16]

    metadata = {
        "version": "v1",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "features": FEATURES,
        "n_train_rows": int(len(train_df)),
        "n_holdout_rows": int(len(hold_df)),
        "train_seconds": train_seconds,
        "model_hash": model_hash,
        "feature_importances": importances,
        "offline_metrics": {
            "model_ndcg@10": round(model_ndcg, 4),
            "model_precision@5": round(model_p5, 4),
            "baseline_ndcg@10": round(base_ndcg, 4),
            "baseline_precision@5": round(base_p5, 4),
            "ndcg_lift_vs_baseline_pct": round(100 * (model_ndcg - base_ndcg) / base_ndcg, 2),
        },
        "known_offline_online_gap_note": (
            "Offline nDCG measures ranking quality on logged impressions only; it does not "
            "capture that better ranking changes WHICH candidates get impressions at all "
            "(position/selection bias). Must be confirmed with an online A/B before claiming a win."
        ),
    }
    with open(f"{REGISTRY_DIR}/v1/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(Path(__file__).resolve().parent / "experiment_log.md", "w") as f:
        f.write("# Experiment Log\n\n")
        f.write(f"- Run at: {metadata['trained_at']}\n")
        f.write(f"- Train rows: {metadata['n_train_rows']}, Holdout rows (unseen jobs): {metadata['n_holdout_rows']}\n")
        f.write(f"- Model: GradientBoostingRegressor(n_estimators=150, max_depth=3, lr=0.08), trained in {train_seconds}s\n")
        f.write(f"- Model hash: {model_hash} (bound to API version v1 — see versioning.py)\n\n")
        f.write("## Offline metrics (held-out jobs, not tuned on)\n\n")
        f.write(f"| metric | baseline | model | lift |\n|---|---|---|---|\n")
        f.write(f"| nDCG@10 | {base_ndcg:.4f} | {model_ndcg:.4f} | {metadata['offline_metrics']['ndcg_lift_vs_baseline_pct']}% |\n")
        f.write(f"| precision@5 | {base_p5:.4f} | {model_p5:.4f} | {round(100*(model_p5-base_p5)/base_p5,2)}% |\n\n")
        f.write("## Feature importances (internal only — never returned raw via API)\n\n")
        for k, v in sorted(importances.items(), key=lambda x: -x[1]):
            f.write(f"- {k}: {v}\n")
        f.write("\n## Honest caveat\n\n" + metadata["known_offline_online_gap_note"] + "\n")

    print(json.dumps(metadata["offline_metrics"], indent=2))
    print("Saved model v1 to registry with hash", model_hash)

    # Train v2 model (intentionally different configuration for version isolation)
    model_v2 = GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42
    )
    t0_v2 = time.time()
    model_v2.fit(train_df[FEATURES], train_df["label"])
    train_seconds_v2 = round(time.time() - t0_v2, 3)

    hold_df["model_score_v2"] = model_v2.predict(hold_df[FEATURES])
    model_ndcg_v2, model_p5_v2 = eval_per_job(hold_df, "model_score_v2")

    importances_v2 = dict(zip(FEATURES, [round(float(x), 4) for x in model_v2.feature_importances_]))
    
    os.makedirs(f"{REGISTRY_DIR}/v2", exist_ok=True)
    with open(f"{REGISTRY_DIR}/v2/model.pkl", "wb") as f:
        pickle.dump(model_v2, f)
        
    model_hash_v2 = hashlib.sha256(open(f"{REGISTRY_DIR}/v2/model.pkl", "rb").read()).hexdigest()[:16]

    metadata_v2 = {
        "version": "v2",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "features": FEATURES,
        "n_train_rows": int(len(train_df)),
        "n_holdout_rows": int(len(hold_df)),
        "train_seconds": train_seconds_v2,
        "model_hash": model_hash_v2,
        "feature_importances": importances_v2,
        "offline_metrics": {
            "model_ndcg@10": round(model_ndcg_v2, 4),
            "model_precision@5": round(model_p5_v2, 4),
            "baseline_ndcg@10": round(base_ndcg, 4),
            "baseline_precision@5": round(base_p5, 4),
            "ndcg_lift_vs_baseline_pct": round(100 * (model_ndcg_v2 - base_ndcg) / base_ndcg, 2),
        },
        "known_offline_online_gap_note": (
            "Offline nDCG measures ranking quality on logged impressions only; it does not "
            "capture that better ranking changes WHICH candidates get impressions at all "
            "(position/selection bias). Must be confirmed with an online A/B before claiming a win."
        ),
    }
    with open(f"{REGISTRY_DIR}/v2/metadata.json", "w") as f:
        json.dump(metadata_v2, f, indent=2)

    print("Saved model v2 to registry with hash", model_hash_v2)


if __name__ == "__main__":
    main()
