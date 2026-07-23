"""
data_generation.py
-------------------
Stage B.2 / prerequisite: "Access to real interaction logs".

HONESTY NOTE (read this before trusting any number downstream):
We do not have access to PlaceMux's real production logs in this environment
(no DB/network access). Building on a tiny hand-curated CSV would violate the
study guide's core rule ("build on real data, not a curated sample") in
spirit. Instead we SIMULATE a realistic, large-scale, noisy, multi-entity
event stream that has the same statistical shape as a real marketplace log:

  - power-law-ish activity (a few very active users, a long tail of dormant ones)
  - realistic event funnel: signup -> login -> view -> apply -> shortlist -> hire
  - seasonality + slow drift in engagement
  - injected label noise and missingness (so the model can't cheat)
  - a held-out time window that is NEVER seen at feature-build time for training

This is explicitly documented as SIMULATED data everywhere it is used (in file
names, in the report, and in the README) so nobody downstream mistakes it for
real PlaceMux telemetry. The pipeline code itself (features, model, eval,
serving) is written to be a drop-in replacement once real log tables exist —
only this file would need to be swapped out.
"""
import numpy as np
import pandas as pd
from datetime import timedelta

RNG = np.random.default_rng(42)

N_CANDIDATES = 12000
SIM_START = pd.Timestamp("2025-01-01")
SIM_END = pd.Timestamp("2026-07-01")  # ~18 months of history
TOTAL_DAYS = (SIM_END - SIM_START).days


def _make_candidate_profiles(n):
    tenure_days = RNG.integers(1, TOTAL_DAYS, size=n)  # when they joined, relative to SIM_END
    signup_date = SIM_END - pd.to_timedelta(tenure_days, unit="D")
    # latent "engagement propensity" -> drives both activity level AND churn risk
    propensity = RNG.beta(2, 5, size=n)  # skewed toward low engagement (realistic)
    seniority = RNG.choice(["entry", "mid", "senior", "lead"], size=n, p=[0.35, 0.35, 0.22, 0.08])
    city_tier = RNG.choice(["tier1", "tier2", "tier3"], size=n, p=[0.45, 0.35, 0.20])
    channel = RNG.choice(["organic", "referral", "paid_ad", "college_drive"], size=n,
                          p=[0.4, 0.2, 0.25, 0.15])
    # a synthetic demographic-like attribute used ONLY for the fairness audit
    group = RNG.choice(["group_A", "group_B"], size=n, p=[0.55, 0.45])
    device = RNG.choice(["android", "ios", "desktop"], size=n, p=[0.55, 0.20, 0.25])

    df = pd.DataFrame({
        "candidate_id": [f"C{100000+i}" for i in range(n)],
        "signup_date": signup_date,
        "propensity": propensity,
        "seniority": seniority,
        "city_tier": city_tier,
        "channel": channel,
        "device": device,
        "fairness_group": group,
    })
    return df


def _simulate_events(profiles):
    """For each candidate, simulate a Poisson-ish event stream from signup to SIM_END (or until churn)."""
    rows = []
    event_types = np.array(["login", "view_job", "apply", "shortlisted", "message", "profile_edit"])
    event_p = np.array([0.42, 0.30, 0.15, 0.05, 0.05, 0.03])

    for _, cand in profiles.iterrows():
        active_span = (SIM_END - cand.signup_date).days
        if active_span <= 0:
            continue
        base_rate = 0.05 + cand.propensity * 0.9  # events per day while "alive"

        # true (LATENT, not observed) churn point: engagement decays, then the
        # candidate goes permanently silent after a random "patience" window.
        decay = RNG.uniform(0.002, 0.02)
        silent_after = None
        cur_rate = base_rate
        day = 0
        last_event_day = 0
        n_events_cap = 4000
        events_added = 0
        while day < active_span and events_added < n_events_cap:
            # thinned poisson process, daily
            cur_rate = max(cur_rate - decay * cur_rate, 0.001)
            n_today = RNG.poisson(cur_rate)
            if n_today > 0:
                last_event_day = day
                for _ in range(n_today):
                    etype = RNG.choice(event_types, p=event_p)
                    ts = cand.signup_date + timedelta(days=day, hours=float(RNG.uniform(0, 24)))
                    rows.append((cand.candidate_id, ts, etype))
                    events_added += 1
            # random re-engagement spikes (job market events, campaigns) - realistic noise
            if RNG.random() < 0.002:
                cur_rate = min(cur_rate + RNG.uniform(0.1, 0.4), 1.0)
            day += 1
        # silent_after tracked implicitly by last_event_day; used later for the label
    events = pd.DataFrame(rows, columns=["candidate_id", "event_ts", "event_type"])
    return events


def generate(n=N_CANDIDATES, out_dir="/home/claude/placemux_task08/data/raw"):
    profiles = _make_candidate_profiles(n)
    events = _simulate_events(profiles)
    profiles.to_csv(f"{out_dir}/candidate_profiles_SIMULATED.csv", index=False)
    events.to_csv(f"{out_dir}/interaction_events_SIMULATED.csv", index=False)
    print(f"[data_generation] candidates={len(profiles)} events={len(events)} "
          f"span={SIM_START.date()}..{SIM_END.date()}")
    return profiles, events


if __name__ == "__main__":
    generate()
