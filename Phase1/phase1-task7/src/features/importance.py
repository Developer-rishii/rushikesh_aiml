"""
importance.py — Step 4: train a model and inspect feature importance,
via permutation importance on VALIDATION data (never training data —
training-data importance reflects overfitting, not generalizable signal).
"""
import logging
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, make_scorer

log = logging.getLogger("src.features.importance")


def compute_permutation_importance(model, X_val: pd.DataFrame, y_val, cfg) -> pd.DataFrame:
    scorer = make_scorer(average_precision_score, response_method="predict_proba")
    result = permutation_importance(
        model, X_val, y_val,
        scoring=scorer,
        n_repeats=cfg.n_permutation_repeats,
        random_state=cfg.seed,
    )
    df = pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    log.info("[Step 4] Computed permutation importance for %s features (top: %s)",
              len(df), df.iloc[0]["feature"])
    return df
