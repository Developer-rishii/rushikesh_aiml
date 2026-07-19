"""
Trains the candidate<->job ranking model and produces a versioned,
reproducible model artifact.

Design decisions (see docs/EXPERIMENT_LOG / alternative approaches):
- Objective: POINTWISE learning-to-rank (regress graded relevance label
  0..3, rank by predicted score) using GradientBoostingRegressor.
  REJECTED: full pairwise/listwise LambdaMART (e.g. LightGBM's
  lambdarank objective) -- not installable in this offline sandbox
  (no network egress for pip). Documented as a known gap; the code path
  is written so the model class is a single swappable component
  (see `build_model()`) and LambdaMART could be dropped in later.
- Split: TIME-BASED (train on days 0-23, held-out test on days 24-29),
  NOT random, because random splits leak future information (a candidate
  or job appearing in both train and test lets the model memorize
  identity rather than learn generalizable matching signal) and because
  real serving traffic is always in the future relative to training data.
- Baseline: popularity ranker (rank purely by job_popularity) -- the
  simplest thing that could possibly work. The model must beat this
  baseline on held-out nDCG@10, not just do "okay" in isolation.
"""
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.feature_pipeline import compute_features_batch, FEATURE_COLUMNS, FEATURE_PIPELINE_VERSION

ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "models"
EXPERIMENT_LOG = Path(__file__).resolve().parents[2] / "experiments" / "experiment_log.md"


def ndcg_at_k(labels_true, scores_pred, k=10):
    """nDCG@k for one query (one candidate's list of job impressions)."""
    order = np.argsort(-scores_pred)[:k]
    gains = (2 ** np.array(labels_true)[order] - 1)
    discounts = np.log2(np.arange(2, len(gains) + 2))
    dcg = np.sum(gains / discounts)

    ideal_order = np.argsort(-np.array(labels_true))[:k]
    ideal_gains = (2 ** np.array(labels_true)[ideal_order] - 1)
    ideal_discounts = np.log2(np.arange(2, len(ideal_gains) + 2))
    idcg = np.sum(ideal_gains / ideal_discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def mean_ndcg(df, score_col, k=10, group_col="candidate_id", label_col="label"):
    scores = []
    for _, g in df.groupby(group_col):
        if len(g) < 2 or g[label_col].sum() == 0:
            continue
        scores.append(ndcg_at_k(g[label_col].values, g[score_col].values, k=k))
    return float(np.mean(scores)) if scores else 0.0


def precision_at_k(df, score_col, k=5, group_col="candidate_id", label_col="label"):
    precisions = []
    for _, g in df.groupby(group_col):
        if len(g) < 2:
            continue
        top_k = g.sort_values(score_col, ascending=False).head(k)
        precisions.append((top_k[label_col] > 0).mean())
    return float(np.mean(precisions)) if precisions else 0.0


def build_model():
    return GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.08,
        subsample=0.8, random_state=42,
    )


def file_sha256(path, n_bytes=2_000_000):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def main():
    root = Path(__file__).resolve().parents[2]
    raw_path = root / "data" / "raw" / "interaction_logs.csv"
    df = pd.read_csv(raw_path)

    # ---- time-based split (no leakage) ----
    train_df = df[df["day_idx"] < 24].copy()
    test_df = df[df["day_idx"] >= 24].copy()
    print(f"Train rows: {len(train_df):,} (days 0-23) | Held-out test rows: {len(test_df):,} (days 24-29)")

    # ---- features via the SAME pipeline serving will use ----
    X_train = compute_features_batch(train_df)
    y_train = train_df["label"].values
    X_test = compute_features_batch(test_df)

    model = build_model()
    model.fit(X_train, y_train)

    # ---- score held-out set, never tuned on ----
    test_df = test_df.copy()
    test_df["model_score"] = model.predict(X_test)
    test_df["baseline_score"] = test_df["job_popularity"]  # naive popularity baseline

    model_ndcg = mean_ndcg(test_df, "model_score", k=10)
    baseline_ndcg = mean_ndcg(test_df, "baseline_score", k=10)
    model_map = precision_at_k(test_df, "model_score", k=5)
    baseline_map = precision_at_k(test_df, "baseline_score", k=5)

    # ---- online-effect proxy: simulate CTR if we had served model's top-1 vs baseline's top-1 ----
    def simulated_ctr(score_col):
        top1 = test_df.sort_values("model_score" if False else score_col, ascending=False)
        top1_per_cand = test_df.loc[test_df.groupby("candidate_id")[score_col].idxmax()]
        return float(top1_per_cand["clicked"].mean())

    model_sim_ctr = simulated_ctr("model_score")
    baseline_sim_ctr = simulated_ctr("baseline_score")

    print(f"\nOFFLINE (held-out, days 24-29, never tuned on):")
    print(f"  nDCG@10   model={model_ndcg:.4f}  baseline(popularity)={baseline_ndcg:.4f}  "
          f"gap={model_ndcg-baseline_ndcg:+.4f}")
    print(f"  P@5       model={model_map:.4f}  baseline(popularity)={baseline_map:.4f}  "
          f"gap={model_map-baseline_map:+.4f}")
    print(f"\nSIMULATED ONLINE PROXY (top-1-shown CTR):")
    print(f"  model={model_sim_ctr:.4f}  baseline={baseline_sim_ctr:.4f}  "
          f"uplift={(model_sim_ctr/max(baseline_sim_ctr,1e-6)-1)*100:+.1f}%")
    print(f"  NOTE: offline nDCG win and online CTR-proxy win must BOTH be positive "
          f"before this ships -- an offline win alone is not sufficient evidence.")

    # ---- feature importances (used for per-prediction explanations at serve time) ----
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4).tolist()))

    # ---- versioned artifact + metadata (model registry, lightweight/local) ----
    version_tag = datetime.utcnow().strftime("model_v%Y%m%d_%H%M%S")
    version_dir = ARTIFACT_ROOT / version_tag
    version_dir.mkdir(parents=True, exist_ok=True)

    import pickle
    with open(version_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "version": version_tag,
        "trained_at_utc": datetime.utcnow().isoformat(),
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "training_data_file": str(raw_path),
        "training_data_sha256_16": file_sha256(raw_path),
        "training_rows": len(train_df),
        "held_out_test_rows": len(test_df),
        "split_strategy": "time-based (train day_idx<24, test day_idx>=24)",
        "model_class": "sklearn.ensemble.GradientBoostingRegressor",
        "hyperparams": model.get_params(),
        "feature_importances": importances,
        "offline_metrics": {
            "ndcg_at_10_model": model_ndcg,
            "ndcg_at_10_baseline_popularity": baseline_ndcg,
            "precision_at_5_model": model_map,
            "precision_at_5_baseline_popularity": baseline_map,
        },
        "online_proxy_metrics": {
            "simulated_ctr_model": model_sim_ctr,
            "simulated_ctr_baseline": baseline_sim_ctr,
        },
        "slo_quality_floor_ndcg_at_10": 0.60,
        "passes_quality_floor": bool(model_ndcg >= 0.60),
    }
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # ---- update "latest" pointer (what serving reads) ----
    with open(ARTIFACT_ROOT / "LATEST_VERSION.txt", "w") as f:
        f.write(version_tag)

    # ---- experiment log (reproducibility) ----
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPERIMENT_LOG, "a") as f:
        f.write(f"\n## {version_tag}\n")
        f.write(f"- trained_at: {metadata['trained_at_utc']}\n")
        f.write(f"- feature_pipeline_version: {FEATURE_PIPELINE_VERSION}\n")
        f.write(f"- training_data_sha256_16: {metadata['training_data_sha256_16']}\n")
        f.write(f"- split: {metadata['split_strategy']}\n")
        f.write(f"- nDCG@10 model={model_ndcg:.4f} vs baseline={baseline_ndcg:.4f} "
                f"(gap {model_ndcg-baseline_ndcg:+.4f})\n")
        f.write(f"- P@5 model={model_map:.4f} vs baseline={baseline_map:.4f}\n")
        f.write(f"- simulated online CTR model={model_sim_ctr:.4f} vs baseline={baseline_sim_ctr:.4f}\n")
        f.write(f"- passes_quality_floor(nDCG@10>=0.60): {metadata['passes_quality_floor']}\n")
        f.write(f"- feature_importances: {importances}\n")

    print(f"\nSaved model artifact -> {version_dir}")
    print(f"Updated LATEST_VERSION.txt -> {version_tag}")
    print(f"Appended experiment record -> {EXPERIMENT_LOG}")
    return version_tag


if __name__ == "__main__":
    main()
