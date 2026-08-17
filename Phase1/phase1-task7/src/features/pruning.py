"""
pruning.py — Step 5: prune useless/leaky features. Two independent gates,
never conflated:

  1. LEAKAGE gate (domain reasoning + statistical smell test, same method
     as Task 2): any feature whose correlation with the target exceeds
     `leakage_corr_threshold` is dropped outright, regardless of how much
     it would help the metric — a feature that "helps" only because it
     leaks the label isn't a real signal.

  2. USEFULNESS gate: among features that pass the leakage gate, any
     whose permutation-importance lift is at or below `min_importance_lift`
     is dropped — it isn't leaking, it's just noise. This directly targets
     the pitfall "Adding features without measuring lift": every engineered
     feature has to earn its place with a measured number, not intuition.
"""
import logging
import pandas as pd

log = logging.getLogger("src.features.pruning")


def check_leakage(df: pd.DataFrame, target_col: str, candidate_cols: list, threshold: float) -> dict:
    y = df[target_col]
    leaky = {}
    for col in candidate_cols:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        corr = df[col].corr(y)
        if pd.notna(corr) and abs(corr) > threshold:
            leaky[col] = round(float(corr), 4)
    if leaky:
        log.warning("[Step 5] Leakage gate flagged %s feature(s): %s", len(leaky), leaky)
    else:
        log.info("[Step 5] Leakage gate: no candidate feature exceeded |corr|>%s", threshold)
    return leaky


def prune_by_importance(importance_df: pd.DataFrame, min_lift: float, protected: set) -> dict:
    """
    protected: features that must never be pruned by the usefulness gate
    (e.g. the original, already-vetted Task 2 baseline features) — this
    task only prunes NEWLY ENGINEERED candidates on usefulness grounds,
    it doesn't second-guess the already-locked prior baseline.
    """
    low_importance = importance_df[
        (importance_df["importance_mean"] <= min_lift) & (~importance_df["feature"].isin(protected))
    ]["feature"].tolist()
    kept = importance_df[~importance_df["feature"].isin(low_importance)]["feature"].tolist()

    log.info("[Step 5] Usefulness gate: dropping %s low-lift feature(s) (<=%.4f): %s",
              len(low_importance), min_lift, low_importance)
    return {"dropped_low_importance": low_importance, "kept": kept}
