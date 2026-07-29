"""
Fairness metrics used across the audit.

We compute BOTH demographic parity and equal opportunity and report
both, because (per the study guide) they conflict and a defensible
audit states which one is optimized for and why:

- Demographic Parity Difference (DPD): P(pred=1|F) - P(pred=1|M)
  "Do groups get shortlisted at the same rate?"
- Equal Opportunity Difference (EOD): TPR(F) - TPR(M), i.e. among
  candidates who ARE truly qualified (label=1), do groups get
  shortlisted at the same rate?

We optimize/report against EQUAL OPPORTUNITY as the primary metric
(see reports/bias_audit_report.md for the justification: in a hiring
context, false negatives for qualified candidates are the harm we
are most legally and ethically exposed on - denying an opportunity
to someone who deserved it - vs. demographic parity, which can force
shortlisting of clearly unqualified candidates and is harder to
defend as "merit-blind" in front of a regulator).
"""
import numpy as np
import pandas as pd


def group_rate(y_pred, group_mask):
    if group_mask.sum() == 0:
        return np.nan
    return y_pred[group_mask].mean()


def true_positive_rate(y_true, y_pred, group_mask):
    mask = group_mask & (y_true == 1)
    if mask.sum() == 0:
        return np.nan
    return y_pred[mask].mean()


def false_positive_rate(y_true, y_pred, group_mask):
    mask = group_mask & (y_true == 0)
    if mask.sum() == 0:
        return np.nan
    return y_pred[mask].mean()


def fairness_report(y_true, y_pred, gender, group_a="F", group_b="M"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    gender = np.asarray(gender)
    mask_a = gender == group_a
    mask_b = gender == group_b

    sr_a, sr_b = group_rate(y_pred, mask_a), group_rate(y_pred, mask_b)
    tpr_a, tpr_b = true_positive_rate(y_true, y_pred, mask_a), true_positive_rate(y_true, y_pred, mask_b)
    fpr_a, fpr_b = false_positive_rate(y_true, y_pred, mask_a), false_positive_rate(y_true, y_pred, mask_b)

    report = {
        "group_a": group_a,
        "group_b": group_b,
        "n_a": int(mask_a.sum()),
        "n_b": int(mask_b.sum()),
        "selection_rate_a": round(float(sr_a), 4),
        "selection_rate_b": round(float(sr_b), 4),
        "demographic_parity_diff": round(float(sr_a - sr_b), 4),
        "demographic_parity_ratio": round(float(sr_a / sr_b), 4) if sr_b else None,
        "tpr_a": round(float(tpr_a), 4),
        "tpr_b": round(float(tpr_b), 4),
        "equal_opportunity_diff": round(float(tpr_a - tpr_b), 4),
        "fpr_a": round(float(fpr_a), 4),
        "fpr_b": round(float(fpr_b), 4),
        "equalized_odds_fpr_diff": round(float(fpr_a - fpr_b), 4),
    }
    return report


# EEOC-style four-fifths rule as a secondary, widely recognized threshold
def four_fifths_pass(report):
    ratio = report["demographic_parity_ratio"]
    return ratio is not None and ratio >= 0.8
