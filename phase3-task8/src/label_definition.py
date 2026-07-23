"""
label_definition.py
--------------------
Stage A.1 / Core concept: "'churned' is a choice (inactive N days); the
choice changes everything." and "predicting churn after it happens is
useless; you need lead time to act."

DECISION (write down WHY, including what we rejected):
  - Label: candidate is CHURNED at snapshot date T if they have ZERO events
    of any type in the window (T, T + HORIZON_DAYS].
  - Horizon: HORIZON_DAYS = 21.
      WHY 21 and not 14 or 30: growth's stated intervention (re-engagement
      email/push + a curated job digest) needs ~1-2 weeks to land and be
      acted on, so a 14-day horizon leaves almost no time to intervene
      before the window closes. 30 days was rejected because candidate
      logins are naturally spiky (job-seekers check in bursts around
      application deadlines) and a 30-day silence window starts confusing
      "churned" with "just applied to 3 jobs last week and is now waiting
      to hear back" -- that's not disengagement, that's normal behaviour.
      21 days is the point where, empirically in the simulated event
      generator's decay curve, the two populations (decaying vs bursty)
      separate most cleanly.
  - Eligibility filter (avoids two failure modes explicitly called out in
    the study guide):
      1. Exclude candidates with < MIN_HISTORY_DAYS (7) of tenure at T.
         Rejected alternative: scoring everyone from day 0. Rejected
         because a brand-new signup with no behaviour yet isn't
         "disengaging" -- there's nothing to disengage from, and forcing a
         prediction here is a vanity model (Stage A pitfall).
      2. Exclude candidates who were ALREADY silent for
         ALREADY_DORMANT_DAYS (60) or more at T.
         WHY: predicting that an already-dead account will stay dead is
         "predicting churn after it happens" -- zero actionability, and it
         would inflate our precision/recall numbers for free. This is the
         single most important anti-leakage / anti-vanity-metric guard in
         this file.
  - Label leakage boundary: features (feature_engineering.py) may only see
    events with event_ts <= T. The label may only see events with
    event_ts in (T, T+HORIZON_DAYS]. These two windows never overlap, and
    every snapshot is built by calling the code paths separately.
"""
import pandas as pd

HORIZON_DAYS = 21
MIN_HISTORY_DAYS = 7
ALREADY_DORMANT_DAYS = 60


def build_labels_for_snapshot(profiles: pd.DataFrame, events: pd.DataFrame,
                               as_of_date: pd.Timestamp) -> pd.DataFrame:
    horizon_end = as_of_date + pd.Timedelta(days=HORIZON_DAYS)

    eligible = profiles[profiles.signup_date <= as_of_date - pd.Timedelta(days=MIN_HISTORY_DAYS)].copy()

    ev_before = events[events.event_ts <= as_of_date]
    last_event = ev_before.groupby("candidate_id").event_ts.max()
    eligible = eligible.merge(last_event.rename("last_event_ts"), on="candidate_id", how="left")
    eligible["days_since_last_event"] = (as_of_date - eligible["last_event_ts"]).dt.days
    eligible["days_since_last_event"] = eligible["days_since_last_event"].fillna(
        (as_of_date - eligible["signup_date"]).dt.days)

    # drop already-dormant accounts (nothing actionable left to predict)
    eligible = eligible[eligible["days_since_last_event"] < ALREADY_DORMANT_DAYS].copy()

    ev_future = events[(events.event_ts > as_of_date) & (events.event_ts <= horizon_end)]
    active_in_future = set(ev_future.candidate_id.unique())

    eligible["churned"] = (~eligible.candidate_id.isin(active_in_future)).astype(int)
    eligible["as_of_date"] = as_of_date
    return eligible[["candidate_id", "as_of_date", "churned"]]


def build_multi_snapshot_labels(profiles, events, snapshot_dates):
    frames = [build_labels_for_snapshot(profiles, events, d) for d in snapshot_dates]
    return pd.concat(frames, ignore_index=True)
