"""
src/health_monitor.py
Computes the model-health report: offline vs online metric gap,
train/serve skew detection, per-version and per-segment breakdown.

Offline metric : nDCG@5 (computed from served_score)
Online metrics : CTR, apply_rate, shortlist_rate
Gap            : online_metric - expected_from_offline_score
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

ROOT    = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


# ── nDCG helpers ───────────────────────────────────────────────────────────────
def _dcg(scores: list) -> float:
    return sum(s / np.log2(i + 2) for i, s in enumerate(scores))

def ndcg_at_k(df: pd.DataFrame, k: int = 5,
               score_col: str = "served_score",
               relevance_col: str = "clicked") -> float:
    """
    Compute nDCG@k across all (student, session) groups.
    Uses 'clicked' as binary relevance ground truth.
    """
    scores = []
    for _, grp in df.groupby("student_id"):
        grp = grp.sort_values(score_col, ascending=False).head(k)
        ideal = sorted(grp[relevance_col].tolist(), reverse=True)
        actual = grp[relevance_col].tolist()
        idcg = _dcg(ideal)
        if idcg == 0:
            continue
        scores.append(_dcg(actual) / idcg)
    return float(np.mean(scores)) if scores else 0.0


def _validate_inputs(pred: pd.DataFrame, inter: pd.DataFrame,
                      train: pd.DataFrame, serve: pd.DataFrame) -> None:
    for name, df, cols in [
        ("prediction_logs", pred,
         {"log_id", "served_score", "offline_score", "model_version", "skew"}),
        ("interaction_logs", inter,
         {"log_id", "clicked", "applied", "shortlisted", "model_version"}),
        ("training_features", train, {"log_id", "skill_gap", "verified_skills"}),
        ("serving_features",  serve, {"log_id", "skill_gap", "verified_skills"}),
    ]:
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
    if pred.empty or inter.empty:
        raise ValueError("prediction_logs or interaction_logs is empty")


def compute_health_report(pred_path: Path = None, inter_path: Path = None,
                           train_path: Path = None, serve_path: Path = None) -> dict:
    if pred_path  is None: pred_path  = ROOT / "data" / "prediction_logs.csv"
    if inter_path is None: inter_path = ROOT / "data" / "interaction_logs.csv"
    if train_path is None: train_path = ROOT / "data" / "training_features.csv"
    if serve_path is None: serve_path = ROOT / "data" / "serving_features.csv"

    pred  = pd.read_csv(pred_path)
    inter = pd.read_csv(inter_path)
    train = pd.read_csv(train_path)
    serve = pd.read_csv(serve_path)

    _validate_inputs(pred, inter, train, serve)

    merged = pred.merge(inter[["log_id", "clicked", "applied", "shortlisted"]],
                         on="log_id", how="inner")

    # ── 1. Offline metric (nDCG@5) ────────────────────────────────────────────
    ndcg_overall = ndcg_at_k(merged, k=5, score_col="served_score",
                               relevance_col="clicked")

    # ── 2. Online metrics (CTR, apply, shortlist) ─────────────────────────────
    ctr          = float(merged["clicked"].mean())
    apply_rate   = float(merged["applied"].mean())
    shortlist    = float(merged["shortlisted"].mean())

    # Expected CTR from offline score (calibration baseline)
    expected_ctr = float(merged["served_score"].mean())
    online_offline_gap = round(ctr - expected_ctr, 4)

    # ── 3. Per-model-version breakdown ────────────────────────────────────────
    version_stats = {}
    for mv, grp in merged.groupby("model_version"):
        v_ndcg = ndcg_at_k(grp, k=5)
        version_stats[mv] = {
            "n":             len(grp),
            "ndcg_at_5":     round(v_ndcg, 4),
            "ctr":           round(grp["clicked"].mean(), 4),
            "apply_rate":    round(grp["applied"].mean(), 4),
            "mean_skew":     round(grp["skew"].mean(), 4),
            "online_offline_gap": round(grp["clicked"].mean()
                                         - grp["served_score"].mean(), 4),
        }

    # ── 4. Train/serve skew detection ─────────────────────────────────────────
    skew_results = {}
    common_feats = list(set(train.columns) & set(serve.columns)
                         - {"log_id", "student_id", "job_id",
                             "match_score_train", "match_score_served",
                             "skill_score_served", "skew"})
    for feat in common_feats:
        if train[feat].dtype in [np.float64, np.int64, float, int]:
            stat, pval = ks_2samp(train[feat].dropna(), serve[feat].dropna())
            skew_results[feat] = {
                "ks_statistic": round(float(stat), 4),
                "p_value":      round(float(pval), 6),
                "skew_detected": pval < 0.05,
            }

    # ── 5. Per-segment CTR gap ────────────────────────────────────────────────
    segment_gap = {}
    for attr in ["college_tier", "region"]:
        if attr not in merged.columns:
            continue
        seg = {}
        for val, grp in merged.groupby(attr):
            seg[str(val)] = {
                "n":          len(grp),
                "ctr":        round(grp["clicked"].mean(), 4),
                "apply_rate": round(grp["applied"].mean(), 4),
                "ndcg_at_5":  round(ndcg_at_k(grp), 4),
            }
        segment_gap[attr] = seg

    report = {
        "summary": {
            "ndcg_at_5_offline":    round(ndcg_overall, 4),
            "ctr_online":           round(ctr, 4),
            "apply_rate_online":    round(apply_rate, 4),
            "shortlist_rate":       round(shortlist, 4),
            "expected_ctr_from_score": round(expected_ctr, 4),
            "online_offline_gap":   online_offline_gap,
            "gap_direction":        "model over-confident" if online_offline_gap < 0
                                    else "model under-confident",
            "total_impressions":    len(merged),
        },
        "by_model_version":  version_stats,
        "train_serve_skew":  skew_results,
        "by_segment":        segment_gap,
    }

    with open(REPORTS / "health_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    return report


if __name__ == "__main__":
    report = compute_health_report()
    s = report["summary"]
    print("── Health Report ──")
    print(f"  nDCG@5 (offline):            {s['ndcg_at_5_offline']}")
    print(f"  CTR (online):                {s['ctr_online']}")
    print(f"  Expected CTR from score:     {s['expected_ctr_from_score']}")
    print(f"  Online/offline gap:          {s['online_offline_gap']}  ({s['gap_direction']})")
    print(f"  Apply rate:                  {s['apply_rate_online']}")
    print(f"\n── Per-version ──")
    for v, vs in report["by_model_version"].items():
        print(f"  {v}: nDCG={vs['ndcg_at_5']}, CTR={vs['ctr']}, "
              f"skew={vs['mean_skew']}, gap={vs['online_offline_gap']}")
    print(f"\n── Train/serve skew ──")
    for feat, res in report["train_serve_skew"].items():
        flag = "⚠️ SKEW" if res["skew_detected"] else "✅ OK"
        print(f"  {feat}: KS={res['ks_statistic']}, p={res['p_value']}  {flag}")
