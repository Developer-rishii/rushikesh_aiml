"""
baseline.py
===========
The baseline every candidate model must beat (Section 11: "Real-data
quality & correctness" and the repeated instruction to "evaluate honestly
against a baseline"). This is deliberately NOT a model: it's the current
recruiter-facing heuristic (rank by skill_match_score alone), which is
what a naive keyword-matching system would do. Any learned ranker that
cannot beat this on held-out nDCG@10 is not shippable.
"""
import numpy as np


class SkillMatchBaseline:
    """Ranks purely by skill_match_score. No fitting required."""

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        # shape-compatible with sklearn's predict_proba: column 1 = score
        s = X["skill_match_score"].values
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return self.predict_proba(X)[:, 1]


def fairness_report(df, y_score, protected_col="gender", top_frac=0.2):
    """Demographic parity and equal-opportunity gap between groups.

    NOTE ON THRESHOLD: LambdaMART outputs an unbounded relevance SCORE, not
    a calibrated probability -- an absolute cutoff like >=0.5 is meaningless
    for this model type (it silently produced 0 "positives" for every group
    when first tried here, which would have made the fairness gate pass
    trivially and vacuously instead of measuring anything real). Instead
    "predicted positive" = being ranked in the top `top_frac` of
    candidates, matching what actually happens at serving time (only the
    top slice is shown to a recruiter). Works identically for the
    calibrated baseline too.

    demographic_parity_diff = |P(top-ranked | group A) - P(top-ranked | group B)|
    equal_opportunity_diff  = |TPR_A - TPR_B|  (computed only where ground truth is available)
    """
    d = df.copy()
    d["_score"] = y_score
    cutoff = d["_score"].quantile(1 - top_frac)
    d["_pred"] = (d["_score"] >= cutoff).astype(int)
    groups = d[protected_col].unique()
    if len(groups) != 2:
        groups = sorted(groups)[:2]
    g_a, g_b = groups[0], groups[1]

    pr_a = d.loc[d[protected_col] == g_a, "_pred"].mean()
    pr_b = d.loc[d[protected_col] == g_b, "_pred"].mean()
    dpd = abs(pr_a - pr_b)

    def tpr(sub):
        pos = sub[sub["shortlisted"] == 1]
        return pos["_pred"].mean() if len(pos) else np.nan

    tpr_a = tpr(d[d[protected_col] == g_a])
    tpr_b = tpr(d[d[protected_col] == g_b])
    eod = abs(tpr_a - tpr_b) if not (np.isnan(tpr_a) or np.isnan(tpr_b)) else None

    return {
        "groups_compared": [str(g_a), str(g_b)],
        "selection_rate": {str(g_a): round(float(pr_a), 4), str(g_b): round(float(pr_b), 4)},
        "demographic_parity_diff": round(float(dpd), 4),
        "true_positive_rate": {
            str(g_a): None if np.isnan(tpr_a) else round(float(tpr_a), 4),
            str(g_b): None if np.isnan(tpr_b) else round(float(tpr_b), 4),
        },
        "equal_opportunity_diff": None if eod is None else round(float(eod), 4),
        "pass_threshold_0_10": (dpd < 0.10) and (eod is None or eod < 0.10),
    }
