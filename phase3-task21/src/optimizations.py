"""
Stage C deliverable: "Optimisations reducing cost per inference/shortlist"

Three optimizations, chosen deliberately (see README section 8 for the
alternative that was rejected and why):
  1. Right-sizing   -- smaller model, CPU instead of GPU where sufficient.
  2. Caching        -- dedupe repeat (job, candidate) scoring requests.
  3. Precompute     -- nightly batch scoring for head (high-traffic) jobs,
                        on-demand only for the long tail.
"""
import numpy as np


def right_sizing_savings(baseline_inferences: int, hardware_before="gpu", hardware_after="cpu"):
    """Most inference does not need a GPU (Section 4). Moving CPU-sufficient
    scoring off GPU is a pure cost win with zero inference count change."""
    return dict(
        inferences=baseline_inferences,
        hardware_before=hardware_before,
        hardware_after=hardware_after,
    )


def caching_savings(df, key_cols=("job_id", "candidate_id"), repeat_window_rows=100_000):
    """
    Real logs contain repeat (job, candidate) score requests (recruiter
    re-opens a job, candidate re-appears in multiple shortlists). Caching
    those removes them from the inference count entirely.
    """
    total = len(df)
    unique_pairs = df.drop_duplicates(subset=list(key_cols))
    cache_hits = total - len(unique_pairs)
    hit_rate = cache_hits / total if total else 0.0
    return dict(
        total_requests=total,
        unique_pairs=len(unique_pairs),
        cache_hits=cache_hits,
        cache_hit_rate=round(hit_rate, 4),
        inferences_after_cache=len(unique_pairs),
    )


def precompute_vs_on_demand(df, job_col="job_id", head_fraction=0.2):
    """
    Precompute nightly for the head (top `head_fraction` of jobs by traffic)
    -- amortized over many viewers, freshness cost = up to 24h staleness.
    On-demand for the long tail -- always fresh, full per-request cost.
    """
    counts = df[job_col].value_counts()
    n_head_jobs = max(1, int(len(counts) * head_fraction))
    head_jobs = counts.index[:n_head_jobs]
    head_traffic = counts.loc[head_jobs].sum()
    total_traffic = counts.sum()
    tail_traffic = total_traffic - head_traffic

    # Precompute cost: score each head job once per night (not per viewer).
    n_nights = 30
    precompute_inferences = n_head_jobs * n_nights
    on_demand_inferences = tail_traffic  # long tail still scored per request

    return dict(
        n_jobs_total=len(counts),
        n_head_jobs=n_head_jobs,
        head_traffic_share=round(head_traffic / total_traffic, 4),
        precompute_inferences_per_month=int(precompute_inferences),
        on_demand_inferences=int(on_demand_inferences),
        inferences_after_precompute=int(precompute_inferences + on_demand_inferences),
        max_staleness_hours=24,
    )
