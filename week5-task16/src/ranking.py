"""
Rec v1 — Learning-to-rank model training, baseline computation, and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report
import joblib
import json
import os
from datetime import datetime
from collections import defaultdict

MODEL_PATH = "src/models/ranker.joblib"
METRICS_PATH = "reports/metrics.json"

FEATURE_COLS = [
    "match_score", "skill_overlap_count", "skill_gap_count",
    "years_exposure_avg", "jd_seniority_level", "verified_skill_count",
    "ai_trust_score",
    # derived
    "skill_gap_ratio", "seniority_match", "trust_weighted_score",
    "college_avg_match_score",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def validate_matching_schema(df: pd.DataFrame) -> None:
    """Fail loudly if the upstream Matching v1 output is malformed."""
    required = [
        "student_id", "college_id", "job_id", "match_score",
        "skill_overlap_count", "skill_gap_count", "years_exposure_avg",
        "jd_seniority_level", "verified_skill_count", "ai_trust_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Matching v1 schema validation FAILED. Missing columns: {missing}"
        )
    print(f"  [OK] Matching v1 schema valid ({len(df)} rows, "
          f"{df['student_id'].nunique()} students, "
          f"{df['college_id'].nunique()} colleges)")


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used by the ranker."""
    df = df.copy()
    df["skill_gap_ratio"] = df["skill_gap_count"] / (df["skill_overlap_count"] + 1e-5)
    df["seniority_match"] = (
        abs(df["jd_seniority_level"] - df["years_exposure_avg"].round()) <= 1
    ).astype(int)
    df["trust_weighted_score"] = df["match_score"] * df["ai_trust_score"]

    # College-level aggregate (computed once; does NOT require cross-college lookup)
    college_avg = (
        df.groupby("college_id")["match_score"]
        .mean()
        .reset_index()
        .rename(columns={"match_score": "college_avg_match_score"})
    )
    df = df.merge(college_avg, on="college_id", how="left")
    return df


# ── metric computation ───────────────────────────────────────────────────────

def _compute_ranking_metrics(df_eval: pd.DataFrame, score_col: str, k: int = 5) -> dict:
    """
    Per-student ranking metrics aggregated across all students in df_eval.
    Students with no positive outcomes are excluded from precision/recall/MRR
    (they can't be meaningfully measured).
    """
    precisions, recalls, mrrs, fprs = [], [], [], []

    for sid, grp in df_eval.groupby("student_id"):
        ranked = grp.sort_values(score_col, ascending=False)
        y = ranked["outcome"].values
        n = len(y)
        total_pos = int(y.sum())
        total_neg = n - total_pos

        if total_pos == 0:
            continue  # no positive signal to measure against

        top_k = y[:k]
        tp_k = int(top_k.sum())
        fp_k = int((top_k == 0).sum())

        precisions.append(tp_k / k)
        recalls.append(tp_k / total_pos)

        # MRR: reciprocal rank of first positive
        first_pos = np.where(y == 1)[0]
        mrrs.append(1.0 / (first_pos[0] + 1) if len(first_pos) > 0 else 0.0)

        # FPR@K: fraction of negatives that appear in top-k
        if total_neg > 0:
            fprs.append(fp_k / total_neg)

    return {
        "precision_at_5": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        "recall_at_5": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "mrr": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
        "fpr_at_5": round(float(np.mean(fprs)), 4) if fprs else 0.0,
        "num_students_evaluated": len(precisions),
    }


# ── training & evaluation ────────────────────────────────────────────────────

def train_and_evaluate():
    print("=" * 60)
    print("Rec v1 — Training Pipeline")
    print("=" * 60)

    # 1. Load & validate
    print("\n[1/5] Loading upstream Matching v1 data...")
    matching_df = pd.read_csv("data/matching_v1_output.csv")
    validate_matching_schema(matching_df)

    outcomes_df = pd.read_csv("data/placement_outcomes.csv")
    print(f"  [OK] Placement outcomes: {len(outcomes_df)} rows "
          f"({outcomes_df['outcome'].sum()} positive, "
          f"{(outcomes_df['outcome']==0).sum()} negative)")

    # 2. Feature engineering
    print("\n[2/5] Engineering features...")
    matching_df = add_derived_features(matching_df)
    print(f"  [OK] Added skill_gap_ratio, seniority_match, trust_weighted_score, "
          f"college_avg_match_score")

    # 3. Merge outcomes for training/eval subset
    df = matching_df.merge(outcomes_df, on=["student_id", "job_id"], how="inner")
    print(f"  [OK] Joined dataset for training: {len(df)} rows, "
          f"{df['student_id'].nunique()} students with known outcomes")

    # 4. Train/test split by student_id
    print("\n[3/5] Training learning-to-rank model (GradientBoosting)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["student_id"]))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    X_train, y_train = train_df[FEATURE_COLS], train_df["outcome"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["outcome"]

    model = GradientBoostingClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.1,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"  [OK] Train accuracy: {train_acc:.4f}")
    print(f"  [OK] Test  accuracy: {test_acc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"  [OK] Model persisted -> {MODEL_PATH}")

    # 5. Evaluate rankings — baseline vs model
    print("\n[4/5] Evaluating ranking quality...")

    test_df["predicted_relevance"] = model.predict_proba(X_test)[:, 1]

    baseline_metrics = _compute_ranking_metrics(test_df, "match_score")
    model_metrics = _compute_ranking_metrics(test_df, "predicted_relevance")

    print(f"\n  {'Metric':<20} {'Baseline':>10} {'Model':>10} {'Delta':>10}")
    print(f"  {'-'*50}")
    for key in ["precision_at_5", "recall_at_5", "mrr", "fpr_at_5"]:
        b, m = baseline_metrics[key], model_metrics[key]
        delta = m - b
        better = "UP" if (delta > 0 and key != "fpr_at_5") or (delta < 0 and key == "fpr_at_5") else ("DN" if delta != 0 else "=")
        print(f"  {key:<20} {b:>10.4f} {m:>10.4f} {delta:>+10.4f} {better}")

    # Segment breakdown
    segments = {}
    for college in test_df["college_id"].unique():
        cdf = test_df[test_df["college_id"] == college]
        segments[college] = {
            "baseline": _compute_ranking_metrics(cdf, "match_score"),
            "model": _compute_ranking_metrics(cdf, "predicted_relevance"),
        }

    test_df["trust_tier"] = np.where(test_df["ai_trust_score"] > 0.7, "high_trust", "low_trust")
    for tier in test_df["trust_tier"].unique():
        tdf = test_df[test_df["trust_tier"] == tier]
        segments[tier] = {
            "baseline": _compute_ranking_metrics(tdf, "match_score"),
            "model": _compute_ranking_metrics(tdf, "predicted_relevance"),
        }

    # Feature importances
    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )

    # Proof-of-ranking-quality: pick a student from the test set
    proof_student = None
    for sid in test_df["student_id"].unique():
        sdf = test_df[test_df["student_id"] == sid]
        if sdf["outcome"].sum() >= 1 and len(sdf) >= 3:
            proof_student = sid
            break

    proof = None
    if proof_student:
        sdf = test_df[test_df["student_id"] == proof_student].copy()
        baseline_order = sdf.sort_values("match_score", ascending=False)[
            ["job_id", "match_score", "outcome"]
        ].to_dict("records")
        model_order = sdf.sort_values("predicted_relevance", ascending=False)[
            ["job_id", "predicted_relevance", "outcome"]
        ].to_dict("records")
        proof = {
            "student_id": proof_student,
            "baseline_ranking": baseline_order,
            "model_ranking": model_order,
        }

    # Experiment log
    experiment_log = {
        "timestamp": datetime.now().isoformat(),
        "model_type": "GradientBoostingClassifier",
        "params": {
            "n_estimators": 150, "max_depth": 4,
            "learning_rate": 0.1, "subsample": 0.8,
        },
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
    }

    results = {
        "baseline": baseline_metrics,
        "model": model_metrics,
        "segments": segments,
        "feature_importances": [[f, round(float(v), 5)] for f, v in importances],
        "proof_of_ranking_quality": proof,
        "experiment_log": experiment_log,
        "matching_v1_stats": {
            "total_rows": len(matching_df),
            "total_students": int(matching_df["student_id"].nunique()),
            "total_colleges": int(matching_df["college_id"].nunique()),
        },
    }

    os.makedirs("reports", exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[5/5] Metrics written -> {METRICS_PATH}")
    print("=" * 60)
    return results


if __name__ == "__main__":
    train_and_evaluate()
