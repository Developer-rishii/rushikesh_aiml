"""
evaluation/threshold.py — Step 3: pick the cost-optimal threshold, same
method as Task 6 (confusion-matrix cells named by clinical meaning, swept
across a threshold grid, minimizing expected cost). Applied here to the
CALIBRATED model's probabilities specifically — thresholding only makes
sense on probabilities that mean what they say.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

log = logging.getLogger("src.evaluation.threshold")


def confusion_at_threshold(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, missed_malignancy, unnecessary_biopsy, tp = cm.ravel()
    return {
        "threshold": round(float(threshold), 4),
        "missed_malignancy": int(missed_malignancy),
        "unnecessary_biopsy": int(unnecessary_biopsy),
        "true_negative": int(tn),
        "true_positive": int(tp),
    }


def select_cost_optimal_threshold(y_true, y_proba, cfg, step: float = 0.01) -> dict:
    grid = np.arange(0.01, 1.0, step)
    rows = []
    for t in grid:
        cm = confusion_at_threshold(y_true, y_proba, t)
        cost = cm["missed_malignancy"] * cfg.cost_false_negative + cm["unnecessary_biopsy"] * cfg.cost_false_positive
        rows.append({**cm, "expected_cost": cost})
    sweep = pd.DataFrame(rows)
    best = sweep.loc[sweep["expected_cost"].idxmin()]
    default = sweep.iloc[(sweep["threshold"] - 0.5).abs().argmin()]

    result = {
        "cost_false_negative": cfg.cost_false_negative,
        "cost_false_positive": cfg.cost_false_positive,
        "default_threshold_0.5": {"threshold": float(default["threshold"]),
                                    "expected_cost": float(default["expected_cost"])},
        "recommended_threshold": {"threshold": float(best["threshold"]),
                                    "expected_cost": float(best["expected_cost"]),
                                    "missed_malignancy": int(best["missed_malignancy"]),
                                    "unnecessary_biopsy": int(best["unnecessary_biopsy"])},
    }
    log.info("[Step 3] Cost-optimal threshold=%.4f (cost=%.1f) vs default 0.5 (cost=%.1f)",
              result["recommended_threshold"]["threshold"], result["recommended_threshold"]["expected_cost"],
              result["default_threshold_0.5"]["expected_cost"])
    return result, sweep
