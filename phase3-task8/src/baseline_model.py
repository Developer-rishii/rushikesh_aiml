"""
baseline_model.py
------------------
Brainstorm question from the study guide: "Does the model beat 'they haven't
logged in for 14 days'?" -- so that IS our baseline, verbatim. We deliberately
do NOT pick a fancier baseline, because the point of a baseline is that
anyone (growth, a PM, an auditor) can verify it by hand in one line of SQL.

Alternative considered and REJECTED: full RFM-segmentation rule (Recency +
Frequency + Monetary-equivalent applies, each bucketed into quintiles and
combined). We build this too (rfm_score below) and report it as a secondary
baseline for the "alternative approaches" comparison the guide asks for, but
the PRIMARY baseline the model must beat is the plain 14-day rule, because
that's the one a non-ML stakeholder can sanity check without trusting us.
"""
import numpy as np


def rule_14_day_score(feature_df):
    """Continuous risk score version of the 14-day rule, so we can draw a PR
    curve for it too (not just a single binary point). Risk grows with days
    since last event; this is monotonic with the binary rule at threshold=14."""
    return feature_df["days_since_last_event"].clip(lower=0).to_numpy(dtype=float)


def rule_14_day_binary(feature_df, threshold=14):
    return (feature_df["days_since_last_event"] >= threshold).astype(int)


def rfm_score(feature_df):
    """Secondary baseline: Recency + Frequency combined rule (rejected as the
    PRIMARY baseline for the reason above, but kept for the alternative-
    approaches comparison)."""
    recency = 1.0 / (1.0 + feature_df["days_since_last_event"])
    frequency = feature_df["events_30d"] + feature_df["apply_30d"] * 2
    freq_norm = frequency / (frequency.max() + 1e-9)
    rfm = 0.5 * recency + 0.5 * freq_norm
    return -rfm.to_numpy()  # invert so higher = more at risk, consistent with other scores
