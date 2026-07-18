"""
fairness_metrics.py
Computes group-level fairness metrics from recommendation data.
This is the baseline audit — no ML, pure measurement.

Metrics computed:
  - Demographic parity gap: |P(rec=1|groupA) - P(rec=1|groupB)|
  - Equal opportunity gap:  |recall(groupA) - recall(groupB)|   [where skill_rec=1 is positive]
  - Disparate impact ratio: min_group_rate / max_group_rate  (< 0.80 triggers 4/5ths rule)
  - Mean match score gap:   diff in avg match_score across groups
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


def _group_stats(df: pd.DataFrame, group_col: str,
                 pred_col: str = "production_recommended",
                 truth_col: str = "recommended") -> pd.DataFrame:
    """Per-group: recommendation rate, skill-based rate, recall, match_score mean."""
    rows = []
    for grp, gdf in df.groupby(group_col):
        n = len(gdf)
        rec_rate   = gdf[pred_col].mean()
        skill_rate = gdf[truth_col].mean()
        # Equal opportunity: among skill-positive cases, how many did we recommend?
        pos = gdf[gdf[truth_col] == 1]
        recall = pos[pred_col].mean() if len(pos) > 0 else float("nan")
        rows.append({
            "group":            str(grp),
            "n_pairs":          n,
            "rec_rate":         round(rec_rate,   4),
            "skill_rec_rate":   round(skill_rate, 4),
            "recall":           round(recall,     4) if not np.isnan(recall) else None,
            "mean_match_score": round(gdf["match_score_biased"].mean(), 4),
        })
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def compute_fairness_report(df: pd.DataFrame,
                             protected_attrs: list = None) -> Dict[str, Any]:
    """
    Returns a dict of fairness metrics for each protected attribute.
    df must have columns: college_tier, region, production_recommended,
                          recommended (skill-only), match_score_biased.
    """
    if protected_attrs is None:
        protected_attrs = ["college_tier", "region"]

    report = {}
    for attr in protected_attrs:
        stats = _group_stats(df, attr)
        rates = stats["rec_rate"].values
        disparate_impact = float(rates.min() / rates.max()) if rates.max() > 0 else 1.0

        max_parity_gap = float(rates.max() - rates.min())

        recall_vals = stats["recall"].dropna().values
        eq_opp_gap  = float(recall_vals.max() - recall_vals.min()) \
                      if len(recall_vals) > 1 else 0.0

        score_gap = float(stats["mean_match_score"].max()
                          - stats["mean_match_score"].min())

        report[attr] = {
            "group_stats":        stats.to_dict(orient="records"),
            "disparate_impact":   round(disparate_impact, 4),
            "max_parity_gap":     round(max_parity_gap, 4),
            "equal_opportunity_gap": round(eq_opp_gap, 4),
            "mean_score_gap":     round(score_gap, 4),
            "fails_4_5ths_rule":  disparate_impact < 0.80,
            "verdict":            "⚠️ BIAS DETECTED" if disparate_impact < 0.80
                                  else "✅ WITHIN THRESHOLD",
        }
    return report


def validate_input(df: pd.DataFrame) -> None:
    required = {"student_id", "college_tier", "region",
                 "production_recommended", "recommended", "match_score_biased"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")
    if df.empty:
        raise ValueError("Input dataframe is empty — cannot run fairness audit.")
    if df["student_id"].nunique() < 10:
        raise ValueError("Fewer than 10 students — sample too small for group-level metrics.")


if __name__ == "__main__":
    import json
    df = pd.read_csv(Path(__file__).parent.parent / "data" / "recommendations.csv")
    validate_input(df)
    report = compute_fairness_report(df)
    for attr, res in report.items():
        print(f"\n── {attr} ──")
        print(f"  Disparate impact:      {res['disparate_impact']:.4f}  "
              f"{'❌ FAILS 4/5ths' if res['fails_4_5ths_rule'] else '✅ OK'}")
        print(f"  Max parity gap:        {res['max_parity_gap']:.4f}")
        print(f"  Equal opportunity gap: {res['equal_opportunity_gap']:.4f}")
        print(f"  Mean score gap:        {res['mean_score_gap']:.4f}")
        print(f"  Verdict:               {res['verdict']}")
