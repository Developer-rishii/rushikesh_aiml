"""
src/defect_ranker.py
Trains a defect-severity classifier on admin-reviewed interaction logs,
then scores ALL unreviewed logs to produce a ranked intelligence-defect list.

Model: LightGBM (or GradientBoosting fallback) binary classifier predicting is_defect.
Output: data/ranked_defects.csv — all logs scored, sorted by predicted defect severity.
"""

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split

ROOT       = Path(__file__).parent.parent
MODEL_PATH = ROOT / "src" / "models" / "defect_classifier.pkl"
EXP_LOG    = ROOT / "reports" / "experiment_log.jsonl"
REPORTS    = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

# Defect categories with estimated user impact weights
DEFECT_IMPACT = {
    "false_positive": 1.5,   # high-ranked irrelevant job → wastes user time
    "false_negative": 2.0,   # relevant job buried → direct opportunity loss
    "skew_induced":   1.8,   # train/serve skew distortion
    "none":           0.0,
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering from prediction + interaction signals."""
    region_map = {"urban": 2, "semi_urban": 1, "rural": 0}
    features = pd.DataFrame({
        "served_score":      df["served_score"].fillna(0.5),
        "offline_score":     df["offline_score"].fillna(0.5),
        "score_delta":       (df["served_score"] - df["offline_score"]).abs(),
        "abs_skew":          df["skew"].abs(),
        "rank_position":     df["rank_position"].fillna(3),
        "rank_reciprocal":   (1.0 / np.log2(df["rank_position"].fillna(3) + 1)),
        "college_tier":      df["college_tier"].fillna(2),
        "region_encoded":    df["region"].map(region_map).fillna(1),
        "model_v1_1":        (df["model_version"] == "v1.1").astype(int),
        "model_v1_2":        (df["model_version"] == "v1.2").astype(int),
        "high_score":        (df["served_score"] > 0.70).astype(int),
        "low_score":         (df["served_score"] < 0.35).astype(int),
    })
    return features


def validate_defect_labels(df: pd.DataFrame) -> None:
    required = {"log_id", "is_defect", "served_score", "offline_score",
                 "skew", "rank_position", "college_tier", "region",
                 "model_version", "user_impact_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"defect_labels.csv missing columns: {missing}")
    if df.empty:
        raise ValueError("defect_labels.csv is empty")
    if df["is_defect"].nunique() < 2:
        raise ValueError("defect_labels must contain both defect and non-defect examples")


def train(labels_path: Path = None, pred_path: Path = None) -> dict:
    if labels_path is None:
        labels_path = ROOT / "data" / "defect_labels.csv"
    if pred_path is None:
        pred_path = ROOT / "data" / "prediction_logs.csv"

    labels   = pd.read_csv(labels_path)
    pred_all = pd.read_csv(pred_path)

    validate_defect_labels(labels)

    X = build_features(labels)
    y = labels["is_defect"].astype(int)

    # Split by log_id to prevent leakage
    ids = labels["log_id"].unique()
    tr_ids, te_ids = train_test_split(ids, test_size=0.25, random_state=42)
    tr = labels["log_id"].isin(tr_ids)
    te = labels["log_id"].isin(te_ids)

    X_tr, y_tr = X[tr], y[tr]
    X_te, y_te = X[te], y[te]

    # Always use sklearn GradientBoostingClassifier so the saved .pkl is portable
    # and can be loaded in any environment without requiring lightgbm.
    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.08,
        random_state=42
    )

    clf.fit(X_tr, y_tr)

    # ── Threshold tuning on val ────────────────────────────────────────────────
    val_ids, te_ids2 = train_test_split(te_ids, test_size=0.5, random_state=42)
    val  = labels["log_id"].isin(val_ids)
    te2  = labels["log_id"].isin(te_ids2)
    X_val, y_val = X[val], y[val]
    X_te2, y_te2 = X[te2], y[te2]

    val_proba = clf.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.80, 0.05):
        p = (val_proba >= t).astype(int)
        f = f1_score(y_val, p, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t

    # ── Evaluate on held-out test ──────────────────────────────────────────────
    te_proba = clf.predict_proba(X_te2)[:, 1]
    te_preds = (te_proba >= best_t).astype(int)

    prec = precision_score(y_te2, te_preds, zero_division=0)
    rec  = recall_score(y_te2,    te_preds, zero_division=0)
    f1   = f1_score(y_te2,        te_preds, zero_division=0)
    auc  = roc_auc_score(y_te2,   te_proba) if y_te2.nunique() > 1 else 0.5
    cm   = confusion_matrix(y_te2, te_preds)
    fpr  = float(cm[0,1]/cm[0].sum()) if cm[0].sum() > 0 else 0.0

    # ── Feature importances ────────────────────────────────────────────────────
    try:
        importances = dict(zip(X.columns.tolist(),
                                clf.feature_importances_.round(4).tolist()))
    except AttributeError:
        importances = {}

    # ── Score ALL prediction logs → ranked defect list ────────────────────────
    X_all    = build_features(pred_all)
    all_proba = clf.predict_proba(X_all)[:, 1]

    inter = pd.read_csv(ROOT / "data" / "interaction_logs.csv")
    ranked = pred_all.copy()
    ranked["defect_probability"] = all_proba.round(4)
    ranked["is_predicted_defect"] = (all_proba >= best_t).astype(int)

    # User impact score = defect_prob × rank_reciprocal × (1 + skew penalty)
    rank_rec  = 1.0 / np.log2(ranked["rank_position"] + 1)
    skew_pen  = ranked["skew"].abs() * 2
    ranked["estimated_user_impact"] = (
        all_proba * rank_rec * (1 + skew_pen)
    ).round(4)

    # Merge click signal for context
    ranked = ranked.merge(
        inter[["log_id", "clicked", "applied"]].drop_duplicates("log_id"),
        on="log_id", how="left"
    )

    # Defect category assignment
    ranked["defect_category"] = "none"
    ranked.loc[
        (ranked["is_predicted_defect"] == 1) &
        (ranked["served_score"] > 0.70) & (ranked["clicked"].fillna(0) == 0),
        "defect_category"
    ] = "false_positive"
    ranked.loc[
        (ranked["is_predicted_defect"] == 1) &
        (ranked["served_score"] < 0.40) & (ranked["clicked"].fillna(0) == 1),
        "defect_category"
    ] = "false_negative"
    ranked.loc[
        (ranked["is_predicted_defect"] == 1) &
        (ranked["abs_skew"] if "abs_skew" in ranked.columns
         else ranked["skew"].abs() > 0.05),
        "defect_category"
    ] = ranked.loc[
        (ranked["is_predicted_defect"] == 1),
        "defect_category"
    ].replace("none", "skew_induced")

    ranked_sorted = ranked.sort_values("estimated_user_impact",
                                        ascending=False).reset_index(drop=True)
    ranked_sorted["defect_rank"] = ranked_sorted.index + 1
    ranked_sorted.to_csv(ROOT / "data" / "ranked_defects.csv", index=False)

    # ── Defect summary by category ────────────────────────────────────────────
    defect_summary = (
        ranked_sorted[ranked_sorted["is_predicted_defect"] == 1]
        .groupby("defect_category")
        .agg(count=("log_id", "count"),
              mean_impact=("estimated_user_impact", "mean"),
              mean_rank=("defect_rank", "mean"))
        .round(4).to_dict(orient="index")
    )

    # ── Save model ─────────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"model": clf, "threshold": best_t,
                  "feature_cols": X.columns.tolist()}, MODEL_PATH)

    result = {
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "n_labeled":          len(labels),
        "n_defects_labeled":  int(y.sum()),
        "threshold":          round(best_t, 2),
        "precision":          round(prec, 4),
        "recall":             round(rec,  4),
        "f1":                 round(f1,   4),
        "auc_roc":            round(auc,  4),
        "fpr":                round(fpr,  4),
        "feature_importances":importances,
        "model_path":         str(MODEL_PATH),
        "n_defects_in_all_logs": int(ranked_sorted["is_predicted_defect"].sum()),
        "defect_summary":     defect_summary,
    }

    EXP_LOG.parent.mkdir(exist_ok=True)
    with open(EXP_LOG, "a") as fh:
        fh.write(json.dumps(result, default=str) + "\n")

    return result


def score_one(log_id: str, pred_df: pd.DataFrame = None) -> dict:
    """Score one prediction log entry for defect probability."""
    artifact = joblib.load(MODEL_PATH)
    clf, threshold, feat_cols = (artifact["model"], artifact["threshold"],
                                  artifact["feature_cols"])
    if pred_df is None:
        pred_df = pd.read_csv(ROOT / "data" / "prediction_logs.csv")
    row = pred_df[pred_df["log_id"] == log_id]
    if row.empty:
        return {"error": f"log_id '{log_id}' not found"}
    X  = build_features(row)[feat_cols]
    p  = float(clf.predict_proba(X)[0, 1])
    verdict = "⚠️ DEFECT" if p >= threshold else "✅ OK"
    try:
        imps = dict(zip(feat_cols, clf.feature_importances_))
    except AttributeError:
        imps = {}
    top3 = sorted(imps.items(), key=lambda x: -x[1])[:3]
    reason = (
        f"{verdict} — defect_prob={p:.3f} (threshold={threshold}). "
        f"served_score={row['served_score'].iloc[0]:.3f}, "
        f"skew={row['skew'].iloc[0]:.3f}, "
        f"rank={row['rank_position'].iloc[0]}. "
        f"Top drivers: {', '.join(f'{k}({v:.3f})' for k,v in top3)}."
    )
    return {
        "log_id":           log_id,
        "defect_probability": round(p, 4),
        "verdict":          verdict,
        "model_version":    row["model_version"].iloc[0],
        "reason":           reason,
    }


if __name__ == "__main__":
    print("Training defect classifier...")
    result = train()
    print(f"  Precision: {result['precision']}  Recall: {result['recall']}  "
          f"F1: {result['f1']}  AUC: {result['auc_roc']}  FPR: {result['fpr']}")
    print(f"  Threshold: {result['threshold']}")
    print(f"  Defects predicted in full log: {result['n_defects_in_all_logs']}")
    print(f"  Top features:")
    for k, v in sorted(result["feature_importances"].items(),
                        key=lambda x: -x[1])[:4]:
        print(f"    {k}: {v}")
    print(f"\nDefect summary:")
    for cat, stats in result["defect_summary"].items():
        print(f"  {cat}: count={stats['count']}, "
              f"mean_impact={stats['mean_impact']:.3f}")
