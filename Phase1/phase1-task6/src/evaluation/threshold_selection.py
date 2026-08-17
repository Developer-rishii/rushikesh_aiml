"""
evaluation/threshold_selection.py — Step 4/6: pick a threshold tied to the
REAL cost of each error type, not the default 0.5. This is the direct
answer to the brief's pitfall "Default threshold with no cost reasoning."

Method: sweep a grid of thresholds, compute expected cost at each
(missed_malignancy * cost_false_negative + unnecessary_biopsy *
cost_false_positive), pick the threshold that minimizes total cost on
the validation set. 0.5 is reported alongside purely as a reference
point, never as the answer.
"""
import logging
import numpy as np
import pandas as pd

from src.evaluation.decision_metrics import confusion_matrix_at_threshold

log = logging.getLogger("src.evaluation.threshold_selection")


def sweep_thresholds(y_true, y_proba, cfg) -> pd.DataFrame:
    grid = np.arange(0.01, 1.0, cfg.threshold_grid_step)
    rows = []
    for t in grid:
        cm = confusion_matrix_at_threshold(y_true, y_proba, t)
        expected_cost = (
            cm["missed_malignancy"] * cfg.cost_false_negative
            + cm["unnecessary_biopsy"] * cfg.cost_false_positive
        )
        rows.append({**cm, "expected_cost": expected_cost})
    return pd.DataFrame(rows)


def select_cost_optimal_threshold(y_true, y_proba, cfg) -> dict:
    sweep = sweep_thresholds(y_true, y_proba, cfg)
    best_row = sweep.loc[sweep["expected_cost"].idxmin()]
    default_row = sweep.iloc[(sweep["threshold"] - 0.5).abs().argmin()]

    result = {
        "cost_false_negative_per_case": cfg.cost_false_negative,
        "cost_false_positive_per_case": cfg.cost_false_positive,
        "cost_reasoning": (
            "A missed_malignancy (predicting benign when actually malignant) "
            f"costs {cfg.cost_false_negative}x an unnecessary_biopsy "
            "(predicting malignant when actually benign), because a missed "
            "cancer diagnosis is far more harmful than one extra follow-up test."
        ),
        "default_threshold_0.5": {
            "threshold": round(float(default_row["threshold"]), 4),
            "missed_malignancy": int(default_row["missed_malignancy"]),
            "unnecessary_biopsy": int(default_row["unnecessary_biopsy"]),
            "expected_cost": float(default_row["expected_cost"]),
        },
        "recommended_threshold": {
            "threshold": round(float(best_row["threshold"]), 4),
            "missed_malignancy": int(best_row["missed_malignancy"]),
            "unnecessary_biopsy": int(best_row["unnecessary_biopsy"]),
            "expected_cost": float(best_row["expected_cost"]),
        },
        "cost_saved_vs_default": round(
            float(default_row["expected_cost"] - best_row["expected_cost"]), 4
        ),
    }
    log.info("Threshold selection: default(0.5)=%s cost=%s | recommended=%s cost=%s | saved=%s",
              result["default_threshold_0.5"]["threshold"], result["default_threshold_0.5"]["expected_cost"],
              result["recommended_threshold"]["threshold"], result["recommended_threshold"]["expected_cost"],
              result["cost_saved_vs_default"])
    return result, sweep
