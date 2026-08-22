"""
evaluation/stability.py — Step 4: evaluate across folds and key segments
for stability/fairness. Directly guards against "hidden per-segment
failure" by computing per-segment recall at the chosen threshold and
flagging anything below a configured minimum, rather than reporting one
aggregate number that could hide a weak segment.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

log = logging.getLogger("src.evaluation.stability")


def cross_validate_stability(pipeline_builder, X, y, cfg) -> dict:
    """Re-fits fresh (unfitted) pipelines per fold via pipeline_builder()
    — a callable with no args returning a new unfitted Pipeline — so no
    fold reuses a model fit on a different fold's data."""
    cv = StratifiedKFold(n_splits=cfg.eval_cv_folds, shuffle=True, random_state=cfg.seed)
    scores = []
    from sklearn.metrics import average_precision_score
    for train_idx, test_idx in cv.split(X, y):
        pipe = pipeline_builder()
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
        scores.append(average_precision_score(y.iloc[test_idx], proba))

    scores = np.array(scores)
    result = {
        "cv_folds": cfg.eval_cv_folds,
        "fold_scores": [round(float(s), 4) for s in scores],
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "stable": bool(scores.std() < 0.05),
    }
    log.info("[Step 4] CV stability across %s folds: mean=%.4f std=%.4f (%s)",
              cfg.eval_cv_folds, result["mean"], result["std"], "STABLE" if result["stable"] else "UNSTABLE")
    return result


def evaluate_segments(X, y, y_proba, threshold: float, cfg) -> dict:
    if cfg.segment_feature not in X.columns:
        raise ValueError(f"Segment feature '{cfg.segment_feature}' not found in data columns.")

    segment_col = pd.qcut(X[cfg.segment_feature], q=cfg.n_segments, labels=cfg.segment_labels, duplicates="drop")
    y_pred = (y_proba >= threshold).astype(int)

    from sklearn.metrics import recall_score, precision_score, accuracy_score
    results = {}
    flagged = []
    for label in cfg.segment_labels:
        mask = (segment_col == label).values
        n = int(mask.sum())
        if n == 0:
            results[label] = {"n": 0, "note": "no rows in this segment for this split"}
            continue
        seg_y, seg_pred = y[mask], y_pred[mask]
        recall = float(recall_score(seg_y, seg_pred, zero_division=0))
        results[label] = {
            "n": n,
            "recall": round(recall, 4),
            "precision": round(float(precision_score(seg_y, seg_pred, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(seg_y, seg_pred)), 4),
        }
        if recall < cfg.min_segment_recall:
            flagged.append(label)

    verdict = {
        "segment_feature": cfg.segment_feature,
        "segments": results,
        "flagged_low_recall_segments": flagged,
        "fairness_confirmed": len(flagged) == 0,
    }
    log.info("[Step 4] Segment check (%s): %s", cfg.segment_feature, verdict)
    return verdict
