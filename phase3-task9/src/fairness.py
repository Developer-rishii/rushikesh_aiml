"""
Fairness check feeding the guardrail (see pitfalls: "A fairness audit done
once, at the end, as a formality" - here it runs every simulated day, not
once, so it can actually halt a variant, not just decorate a report).

Metric: demographic parity gap on the shortlist decision - for each JOB
posting, per day, the top-k ranked *candidates* by model score are
"shortlisted". Gap = |shortlist_rate_A - shortlist_rate_B| across the two
synthetic cohort segments A/B (candidates competing for the same jobs, so
grouping by job_id is what makes the shortlist decision meaningful - a
per-user top-k would trivially select ~everyone, since each candidate only
sees a handful of jobs a day). This is a stand-in for a protected-attribute
fairness check; no real protected attribute is used or stored (see
data/generate_logs.py).
"""
import pandas as pd


def selection_rate(df: pd.DataFrame, k: int = 5) -> pd.Series:
    """Per-segment fraction of candidate impressions shortlisted (top-k by
    score) within their job's daily candidate pool. Vectorized via
    rank-within-group instead of groupby.apply, so no columns are dropped."""
    if len(df) == 0:
        return pd.Series(dtype=float)
    ranks = df.groupby(["job_id", "day"])["score"].rank(method="first", ascending=False)
    in_top_k = ranks <= k
    return in_top_k.groupby(df["segment"]).mean()


def demographic_parity_gap(df: pd.DataFrame, k: int = 5) -> float:
    rates = selection_rate(df, k)
    if len(rates) < 2:
        return 0.0
    return float(rates.max() - rates.min())
