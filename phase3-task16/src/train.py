"""
Stage B/C build script: trains a tenant-scoped ranking/match model.

Bar for "good" (written BEFORE looking at results, per Stage 1 "frame & set the bar"):
  - Baseline = popularity baseline (predict the tenant's global mean applied-rate
    for every candidate-job pair, i.e. "ignore the candidate").
  - Metric that decides it = nDCG@10 and precision@5 on a held-out test split,
    plus ROC-AUC on the binary "applied" label as a sanity metric.
  - Must beat baseline nDCG@10 by a real margin (not noise) to be considered done.
"""
import os
import json
import pickle
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from isolation import TenantDataStore, list_tenants
from features import compute_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
EXP_LOG = os.path.join(BASE_DIR, "experiments", "experiment_log.md")


def ndcg_at_k(y_true, y_score, k=10):
    order = np.argsort(-y_score)[:k]
    gains = y_true[order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal_order = np.argsort(-y_true)[:k]
    ideal_gains = y_true[ideal_order]
    idcg = float(np.sum(ideal_gains * discounts[: len(ideal_gains)]))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(y_true, y_score, k=5):
    order = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[order])) if len(order) else 0.0


def train_for_tenant(tenant_id: str):
    store = TenantDataStore(tenant_id)
    df = store.load_logs()
    cfg = store.load_config()

    X = compute_features(df)
    y = df["applied"].values

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.25, random_state=13, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.08, random_state=7
    )
    model.fit(X_train, y_train)

    # ---- evaluate honestly against baseline on the SAME held-out split ----
    y_pred = model.predict_proba(X_test)[:, 1]
    baseline_pred = np.full_like(y_pred, y_train.mean())  # popularity baseline

    model_auc = roc_auc_score(y_test, y_pred)
    baseline_auc = roc_auc_score(y_test, baseline_pred) if len(set(baseline_pred)) > 1 else 0.5

    model_ndcg = ndcg_at_k(y_test, y_pred, k=10)
    baseline_ndcg = ndcg_at_k(y_test, baseline_pred, k=10)

    model_p5 = precision_at_k(y_test, y_pred, k=5)
    baseline_p5 = precision_at_k(y_test, baseline_pred, k=5)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(BASE_DIR, cfg["model_path"])
    version_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "tenant_id": tenant_id,
            "trained_at": version_tag,
            "feature_columns": list(X.columns),
            "train_rows": len(X_train),
        }, f)

    result = {
        "tenant_id": tenant_id,
        "version": version_tag,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "model_auc": round(float(model_auc), 4),
        "baseline_auc": round(float(baseline_auc), 4),
        "model_ndcg@10": round(float(model_ndcg), 4),
        "baseline_ndcg@10": round(float(baseline_ndcg), 4),
        "model_precision@5": round(float(model_p5), 4),
        "baseline_precision@5": round(float(baseline_p5), 4),
        "gap_offline_expected_online": (
            "Offline nDCG uplift is measured on logged/held-out impressions only; "
            "expected online effect will be SMALLER due to position bias and "
            "exploration deficit in logged data. Recommend online A/B before "
            "full rollout, ramping 5% -> 25% -> 100%."
        ),
    }
    return result, df_test, y_pred, y_test


def append_experiment_log(results):
    os.makedirs(os.path.dirname(EXP_LOG), exist_ok=True)
    lines = [f"\n## Run {datetime.now(timezone.utc).isoformat()}\n"]
    for r in results:
        lines.append(f"### {r['tenant_id']} (model version {r['version']})")
        lines.append(f"- train rows: {r['n_train']}, test rows: {r['n_test']}")
        lines.append(f"- AUC: model={r['model_auc']} vs baseline={r['baseline_auc']}")
        lines.append(f"- nDCG@10: model={r['model_ndcg@10']} vs baseline={r['baseline_ndcg@10']}")
        lines.append(f"- precision@5: model={r['model_precision@5']} vs baseline={r['baseline_precision@5']}")
        lines.append(f"- {r['gap_offline_expected_online']}\n")
    with open(EXP_LOG, "a") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    all_results = []
    for t in list_tenants():
        res, df_test, y_pred, y_test = train_for_tenant(t)
        all_results.append(res)
        print(json.dumps(res, indent=2))
    append_experiment_log(all_results)
    with open(os.path.join(BASE_DIR, "evidence", "metrics_report.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nWrote evidence/metrics_report.json and experiments/experiment_log.md")
