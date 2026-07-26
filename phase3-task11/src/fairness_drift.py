"""
fairness_drift.py
==================
Two things the study guide explicitly flags as pitfalls if skipped:
  1. "A fairness audit done once, at the end, as a formality" -> here it's
     a function called from evaluate.py / run_all.py EVERY run, not a
     one-off notebook cell.
  2. "Drift/monitoring tooling" from the recommended stack -> Population
     Stability Index (PSI) between the training feature distribution and
     a later/serving slice, the standard lightweight drift metric.

`region_group` is a synthetic categorical field (NOT a real protected
attribute) used only to illustrate the mechanism: selection-rate parity
at top-5 across segments. In a real deployment this must be run against
whatever protected/DPDP-relevant segments legal/compliance define.
"""
import numpy as np
import pandas as pd


def selection_rate_parity(df: pd.DataFrame, score_col: str, group_col="region_group", top_k=5):
    rates = {}
    for job_id, g in df.groupby("job_id"):
        g = g.sort_values(score_col, ascending=False).head(top_k)
        for grp, cnt in g[group_col].value_counts().items():
            rates.setdefault(grp, []).append(cnt)
    overall_share = df[group_col].value_counts(normalize=True).to_dict()
    parity = {}
    for grp, counts in rates.items():
        selected_share = np.sum(counts) / (df.job_id.nunique() * top_k)
        parity[grp] = {
            "population_share": round(overall_share.get(grp, 0.0), 3),
            "top5_selection_share": round(float(selected_share), 3),
        }
    return parity


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Rule of thumb: <0.1 stable, 0.1-0.25
    moderate drift (watch), >0.25 significant drift (retrain/investigate)."""
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected, quantiles))
    if len(cuts) < 3:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)
    e_pct = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, None)
    a_pct = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def drift_report(train_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols) -> dict:
    return {c: round(psi(train_df[c].values, current_df[c].values), 4) for c in feature_cols}
