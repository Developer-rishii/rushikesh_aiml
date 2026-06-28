"""
PlaceMux Quality Sign-Off - Evaluation Engine
===============================================
Computes precision, recall, FPR for baseline AND trained model on held-out test split.
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from src.features import build_feature_matrix, FEATURE_COLS
from src.labeling import label_dataset
from src.baseline import run_baseline

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")


def _fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4)

def _safe_metrics(y_true, y_pred):
    p = round(precision_score(y_true, y_pred, zero_division=0), 4)
    r = round(recall_score(y_true, y_pred, zero_division=0), 4)
    fpr = _fpr(y_true, y_pred)
    return {"precision": p, "recall": r, "fpr": fpr, "n": len(y_true)}

def _delta(model_metrics, base_metrics):
    return {
        "precision_delta": round(model_metrics["precision"] - base_metrics["precision"], 4),
        "recall_delta": round(model_metrics["recall"] - base_metrics["recall"], 4),
        "fpr_delta": round(model_metrics["fpr"] - base_metrics["fpr"], 4),
        "n": model_metrics["n"],
    }

def run_evaluation(students: pd.DataFrame, jobs: pd.DataFrame,
                   events: pd.DataFrame) -> dict:
    os.makedirs(REPORT_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "match_model.joblib")
    clf = joblib.load(model_path)

    feat_df = build_feature_matrix(students, jobs, events)
    labels = label_dataset(students, jobs, events)
    feat_df = feat_df.reset_index(drop=True)
    labels = labels.iloc[:len(feat_df)].reset_index(drop=True)
    feat_df["label"] = labels

    test_meta = pd.read_csv(os.path.join(DATA_DIR, "test_split.csv"))
    test_app_ids = set(test_meta["application_id"].tolist())
    test_df = feat_df[feat_df["application_id"].isin(test_app_ids)].copy()

    y_true = test_df["label"].values

    baseline_preds = run_baseline(students, jobs,
                                  events[events["application_id"].isin(test_app_ids)])
    baseline_map = dict(zip(baseline_preds["application_id"], baseline_preds["is_match"]))
    y_base = test_df["application_id"].map(baseline_map).fillna(0).astype(int).values

    X_test = test_df[FEATURE_COLS].values
    y_model = clf.predict(X_test)
    
    event_tier = events[["application_id", "price_tier"]].drop_duplicates() if "price_tier" in events.columns else None
    if event_tier is None:
        job_tier = jobs[["job_id", "price_tier"]].drop_duplicates()
        ev_with_tier = events.merge(job_tier, on="job_id", how="left")
        event_tier = ev_with_tier[["application_id", "price_tier"]].drop_duplicates()

    test_df = test_df.merge(
        event_tier.rename(columns={"price_tier": "event_price_tier"}),
        on="application_id",
        how="left",
    )
    if "event_price_tier" in test_df.columns:
        test_df["tier"] = test_df["event_price_tier"]
    else:
        test_df["tier"] = test_df.apply(
            lambda r: "free" if r.get("price_tier_free", 0) else
                      ("basic" if r.get("price_tier_basic", 0) else "premium"), axis=1
        )

    overall_base = _safe_metrics(y_true, y_base)
    overall_model = _safe_metrics(y_true, y_model)
    overall_delta = _delta(overall_model, overall_base)

    pre_mask = test_df["tier"] == "free"
    post_mask = test_df["tier"].isin(["basic", "premium"])

    def _slice_metrics(mask):
        if mask.sum() == 0:
            empty = {"precision": 0, "recall": 0, "fpr": 0, "n": 0}
            return empty, empty, _delta(empty, empty)
        yt = y_true[mask]
        yb = y_base[mask]
        ym = y_model[mask]
        bm = _safe_metrics(yt, yb)
        mm = _safe_metrics(yt, ym)
        return bm, mm, _delta(mm, bm)

    pre_base, pre_model, pre_delta = _slice_metrics(pre_mask.values)
    post_base, post_model, post_delta = _slice_metrics(post_mask.values)

    tier_breakdown = {}
    for tier in ["free", "basic", "premium"]:
        mask = (test_df["tier"] == tier).values
        if mask.sum() < 2: continue
        tb, tm, td = _slice_metrics(mask)
        tier_breakdown[tier] = {"baseline": tb, "model": tm, "delta": td}

    pay_breakdown = {}
    for status in ["success", "failed", "pending", "refunded"]:
        mask = (test_df["payment_status"] == status).values
        if mask.sum() < 2: continue
        pb, pm, pd_ = _slice_metrics(mask)
        pay_breakdown[status] = {"baseline": pb, "model": pm, "delta": pd_}

    guardrail = {
        "precision_degraded": bool(post_model["precision"] < pre_model["precision"]),
        "recall_degraded": bool(post_model["recall"] < pre_model["recall"]),
        "fpr_increased": bool(post_model["fpr"] > pre_model["fpr"]),
        "pre_precision": pre_model["precision"],
        "post_precision": post_model["precision"],
        "pre_recall": pre_model["recall"],
        "post_recall": post_model["recall"],
        "pre_fpr": pre_model["fpr"],
        "post_fpr": post_model["fpr"],
    }

    model_wins = {
        "precision": bool(overall_model["precision"] > overall_base["precision"]),
        "recall": bool(overall_model["recall"] > overall_base["recall"]),
        "fpr_lower": bool(overall_model["fpr"] < overall_base["fpr"]),
    }

    report = {
        "overall": {"baseline": overall_base, "model": overall_model, "delta": overall_delta},
        "pre_monetization": {"baseline": pre_base, "model": pre_model, "delta": pre_delta},
        "post_monetization": {"baseline": post_base, "model": post_model, "delta": post_delta},
        "by_price_tier": tier_breakdown,
        "by_payment_status": pay_breakdown,
        "guardrail": guardrail,
        "model_vs_baseline": model_wins,
        "test_set_size": len(test_df),
    }

    out_path = os.path.join(REPORT_DIR, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[OK] Evaluation complete on {len(test_df)} test samples -> reports/evaluation_results.json")
    print(f"  |  Overall  baseline P={overall_base['precision']:.3f} R={overall_base['recall']:.3f} FPR={overall_base['fpr']:.3f}")
    print(f"  |  Overall  model    P={overall_model['precision']:.3f} R={overall_model['recall']:.3f} FPR={overall_model['fpr']:.3f}")
    print(f"  |  Pre-mon  model    P={pre_model['precision']:.3f} R={pre_model['recall']:.3f} FPR={pre_model['fpr']:.3f}")
    print(f"  |  Post-mon model    P={post_model['precision']:.3f} R={post_model['recall']:.3f} FPR={post_model['fpr']:.3f}")
    flag = "[!] YES" if any([guardrail["precision_degraded"], guardrail["recall_degraded"], guardrail["fpr_increased"]]) else "[OK] NO"
    print(f"  `  Post-monetization degradation? {flag}")

    return report

def main():
    students = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
    jobs = pd.read_csv(os.path.join(DATA_DIR, "jobs.csv"))
    events = pd.read_csv(os.path.join(DATA_DIR, "monetization_events.csv"))
    run_evaluation(students, jobs, events)

if __name__ == "__main__":
    main()
