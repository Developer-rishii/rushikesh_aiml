"""
feature_engineering.py
-----------------------
Core concept from the study guide: "Train/serve skew ... The single biggest
silent killer: features computed one way in training and another way at
serving."

FIX: there is exactly ONE function, `compute_features_as_of`, that both the
training pipeline (train_model.py) and the serving/scoring pipeline
(at_risk_list.py, failure_simulation.py) import and call. Nobody is allowed
to re-implement a feature "the same way but slightly differently" -- that is
how skew is introduced in real systems. If a feature needs to change, it
changes here once.

All features use ONLY events with event_ts <= as_of_date. This is the
leakage boundary. Nothing after as_of_date may ever touch this function.
"""
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "days_since_signup", "days_since_last_event",
    "events_7d", "events_14d", "events_30d", "events_90d",
    "apply_30d", "apply_90d", "login_30d",
    "distinct_event_types_30d", "recency_trend_ratio",
    "seniority_enc", "city_tier_enc", "channel_enc", "device_enc",
]

_SENIORITY_MAP = {"entry": 0, "mid": 1, "senior": 2, "lead": 3}
_TIER_MAP = {"tier1": 0, "tier2": 1, "tier3": 2}
_CHANNEL_MAP = {"organic": 0, "referral": 1, "paid_ad": 2, "college_drive": 3}
_DEVICE_MAP = {"android": 0, "ios": 1, "desktop": 2}


def compute_features_as_of(profiles: pd.DataFrame, events: pd.DataFrame, as_of_date: pd.Timestamp,
                            candidate_ids=None) -> pd.DataFrame:
    """Compute the exact same feature vector regardless of whether this is
    called from the training pipeline (historical as_of_date) or the live
    scoring pipeline (as_of_date = today)."""
    prof = profiles.copy()
    if candidate_ids is not None:
        prof = prof[prof.candidate_id.isin(candidate_ids)]
    prof = prof[prof.signup_date <= as_of_date].copy()

    ev = events[events.event_ts <= as_of_date]
    if len(ev) == 0:
        ev_by_cand = {}
    else:
        ev_by_cand = dict(tuple(ev.groupby("candidate_id")))

    rows = []
    for _, cand in prof.iterrows():
        cid = cand.candidate_id
        days_since_signup = (as_of_date - cand.signup_date).days
        cev = ev_by_cand.get(cid)

        if cev is None or len(cev) == 0:
            days_since_last_event = days_since_signup  # never engaged
            events_7d = events_14d = events_30d = events_90d = 0
            apply_30d = apply_90d = login_30d = 0
            distinct_30d = 0
            recency_trend_ratio = 0.0
        else:
            last_ts = cev.event_ts.max()
            days_since_last_event = (as_of_date - last_ts).days
            d = (as_of_date - cev.event_ts).dt.days
            events_7d = int((d <= 7).sum())
            events_14d = int((d <= 14).sum())
            events_30d = int((d <= 30).sum())
            events_90d = int((d <= 90).sum())
            apply_30d = int(((d <= 30) & (cev.event_type == "apply")).sum())
            apply_90d = int(((d <= 90) & (cev.event_type == "apply")).sum())
            login_30d = int(((d <= 30) & (cev.event_type == "login")).sum())
            distinct_30d = int(cev.loc[d <= 30, "event_type"].nunique())
            prior_7_14 = int(((d > 7) & (d <= 14)).sum())
            recency_trend_ratio = (events_7d + 1.0) / (prior_7_14 + 1.0)  # >1 = accelerating, <1 = decaying

        rows.append({
            "candidate_id": cid,
            "as_of_date": as_of_date,
            "days_since_signup": days_since_signup,
            "days_since_last_event": days_since_last_event,
            "events_7d": events_7d, "events_14d": events_14d,
            "events_30d": events_30d, "events_90d": events_90d,
            "apply_30d": apply_30d, "apply_90d": apply_90d, "login_30d": login_30d,
            "distinct_event_types_30d": distinct_30d,
            "recency_trend_ratio": recency_trend_ratio,
            "seniority_enc": _SENIORITY_MAP[cand.seniority],
            "city_tier_enc": _TIER_MAP[cand.city_tier],
            "channel_enc": _CHANNEL_MAP[cand.channel],
            "device_enc": _DEVICE_MAP[cand.device],
            "fairness_group": cand.fairness_group,  # carried along, NEVER fed to the model
        })
    return pd.DataFrame(rows)


def sufficient_history_mask(feat_df: pd.DataFrame, min_days=7) -> pd.Series:
    """Edge case: brand-new signups don't have enough history for a meaningful
    score. Flag them instead of forcing a (meaningless) prediction."""
    return feat_df["days_since_signup"] >= min_days
