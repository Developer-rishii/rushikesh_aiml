"""
Out-of-sample validation: load the persisted Rec v1 model and evaluate it
on the fresh dataset it has never seen.  Compare against baseline and
the original training-time metrics from Task 16.
"""
import pandas as pd
import numpy as np
import joblib
import json
import os

# ── Feature columns (must match Task 16's ranking.py) ────────────────────
FEATURE_COLS = [
    "match_score", "skill_overlap_count", "skill_gap_count",
    "years_exposure_avg", "jd_seniority_level", "verified_skill_count",
    "ai_trust_score",
    "skill_gap_ratio", "seniority_match", "trust_weighted_score",
    "college_avg_match_score",
]


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicate Task 16's derived feature engineering."""
    df = df.copy()
    df["skill_gap_ratio"] = df["skill_gap_count"] / (df["skill_overlap_count"] + 1e-5)
    df["seniority_match"] = (
        abs(df["jd_seniority_level"] - df["years_exposure_avg"].round()) <= 1
    ).astype(int)
    df["trust_weighted_score"] = df["match_score"] * df["ai_trust_score"]
    college_avg = (
        df.groupby("college_id")["match_score"]
        .mean()
        .reset_index()
        .rename(columns={"match_score": "college_avg_match_score"})
    )
    df = df.merge(college_avg, on="college_id", how="left")
    return df


def compute_ranking_metrics(df_eval: pd.DataFrame, score_col: str, k: int = 5) -> dict:
    """Per-student ranking metrics (same logic as Task 16)."""
    precisions, recalls, mrrs, fprs = [], [], [], []

    for _, grp in df_eval.groupby("student_id"):
        ranked = grp.sort_values(score_col, ascending=False)
        y = ranked["outcome"].values
        n = len(y)
        total_pos = int(y.sum())
        total_neg = n - total_pos

        if total_pos == 0:
            continue

        top_k = y[:k]
        tp_k = int(top_k.sum())
        fp_k = int((top_k == 0).sum())

        precisions.append(tp_k / k)
        recalls.append(tp_k / total_pos)

        first_pos = np.where(y == 1)[0]
        mrrs.append(1.0 / (first_pos[0] + 1) if len(first_pos) > 0 else 0.0)

        if total_neg > 0:
            fprs.append(fp_k / total_neg)

    return {
        "precision_at_5": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        "recall_at_5": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        "mrr": round(float(np.mean(mrrs)), 4) if mrrs else 0.0,
        "fpr_at_5": round(float(np.mean(fprs)), 4) if fprs else 0.0,
        "num_students_evaluated": len(precisions),
    }


def run_oos_validation(data_dir: str, reports_dir: str, task16_dir: str) -> dict:
    """Run out-of-sample validation pipeline."""
    print("\n[OOS] Loading fresh data...")
    matching_df = pd.read_csv(os.path.join(data_dir, "fresh_matching.csv"))
    outcomes_df = pd.read_csv(os.path.join(data_dir, "fresh_outcomes.csv"))
    students_meta = pd.read_csv(os.path.join(data_dir, "fresh_students_meta.csv"))

    if len(matching_df) < 50:
        raise ValueError(
            f"Insufficient data for validation: only {len(matching_df)} matching rows. "
            "Need at least 50 for meaningful metrics."
        )

    # Feature engineering
    matching_df = add_derived_features(matching_df)

    # Merge outcomes
    df = matching_df.merge(outcomes_df, on=["student_id", "job_id"], how="inner")
    print(f"[OOS] Joined dataset: {len(df)} rows, "
          f"{df['student_id'].nunique()} students with outcomes")

    if len(df) < 30:
        raise ValueError(
            f"Insufficient joined data: only {len(df)} rows after outcome merge. "
            "Cannot produce reliable metrics."
        )

    # Load persisted Rec v1 model
    model_path = os.path.join(task16_dir, "src", "models", "ranker.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Rec v1 model not found at {model_path}. "
            "Ensure week5-task16 has been run first."
        )
    model = joblib.load(model_path)
    print(f"[OOS] Loaded Rec v1 model from {model_path}")

    # Predict (no retraining!)
    X = df[FEATURE_COLS]
    df["predicted_relevance"] = model.predict_proba(X)[:, 1]

    # ── Overall metrics ──────────────────────────────────────────────────
    fresh_model = compute_ranking_metrics(df, "predicted_relevance")
    fresh_baseline = compute_ranking_metrics(df, "match_score")

    # Load original training-time metrics for comparison
    orig_metrics_path = os.path.join(task16_dir, "reports", "metrics.json")
    if os.path.exists(orig_metrics_path):
        with open(orig_metrics_path) as f:
            orig = json.load(f)
        original_model = orig.get("model", {})
        original_baseline = orig.get("baseline", {})
    else:
        original_model = {}
        original_baseline = {}

    # ── Segmented metrics ────────────────────────────────────────────────
    segments = {}

    # By college
    for college in df["college_id"].unique():
        cdf = df[df["college_id"] == college]
        segments[college] = {
            "baseline": compute_ranking_metrics(cdf, "match_score"),
            "model": compute_ranking_metrics(cdf, "predicted_relevance"),
        }

    # By seniority level (merge student metadata)
    df_with_meta = df.merge(
        students_meta[["student_id", "seniority", "college_size"]],
        on="student_id", how="left"
    )
    for level in df_with_meta["seniority"].dropna().unique():
        ldf = df_with_meta[df_with_meta["seniority"] == level]
        segments[f"seniority_{level}"] = {
            "baseline": compute_ranking_metrics(ldf, "match_score"),
            "model": compute_ranking_metrics(ldf, "predicted_relevance"),
        }

    # By college size (new dimension not in Task 16)
    for size in df_with_meta["college_size"].dropna().unique():
        sdf = df_with_meta[df_with_meta["college_size"] == size]
        segments[f"college_size_{size}"] = {
            "baseline": compute_ranking_metrics(sdf, "match_score"),
            "model": compute_ranking_metrics(sdf, "predicted_relevance"),
        }

    results = {
        "fresh_data_model": fresh_model,
        "fresh_data_baseline": fresh_baseline,
        "original_training_model": original_model,
        "original_training_baseline": original_baseline,
        "segments": segments,
        "fresh_data_stats": {
            "matching_rows": len(matching_df),
            "outcome_rows": len(outcomes_df),
            "joined_rows": len(df),
            "students": int(df["student_id"].nunique()),
            "colleges": int(df["college_id"].nunique()),
        },
    }

    os.makedirs(reports_dir, exist_ok=True)
    with open(os.path.join(reports_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OOS] Metrics saved to reports/metrics.json")

    return results


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_oos_validation(
        os.path.join(base, "data"),
        os.path.join(base, "reports"),
        os.path.join(base, "..", "week5-task16"),
    )
