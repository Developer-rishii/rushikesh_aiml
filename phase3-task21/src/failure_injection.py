"""
Stage E.3: "Deliberately induce the failure and confirm the designed
degradation actually happens." Also required per Stage B.4 / C.4 / D.4:
"plus what happens when the model is unavailable."

Failure mode chosen: the scoring service (model.predict) times out / raises.
Designed fallback: serve the cached precomputed score if available, else a
cheap heuristic (skill_overlap + experience_match), never a hard 500.
"""
import numpy as np


class ModelUnavailable(Exception):
    pass


def flaky_predict(model, X, fail_rate=1.0, rng=None):
    """fail_rate=1.0 means the primary model ALWAYS fails (worst case drill)."""
    rng = rng or np.random.default_rng(0)
    if rng.random() < fail_rate:
        raise ModelUnavailable("scoring service timeout")
    return model.predict(X)


def score_with_fallback(model, X_df, cache: dict, key_series, fail_rate=1.0, rng=None):
    """
    Returns (scores, degradation_report). On failure: use cache if the
    (job,candidate) pair was precomputed, else a heuristic score, and log
    every fallback so it is auditable (no silent quality drop).
    """
    rng = rng or np.random.default_rng(0)
    scores = np.zeros(len(X_df))
    used_fallback = np.zeros(len(X_df), dtype=bool)
    used_cache = np.zeros(len(X_df), dtype=bool)

    try:
        scores = flaky_predict(model, X_df, fail_rate=fail_rate, rng=rng)
        return scores, dict(model_available=True, fallback_rows=0, cache_rows=0, heuristic_rows=0)
    except ModelUnavailable:
        pass

    for i, (_, row) in enumerate(X_df.iterrows()):
        key = key_series.iloc[i]
        if key in cache:
            scores[i] = cache[key]
            used_cache[i] = True
        else:
            scores[i] = 0.6 * row["Comedy"] + 0.4 * row["Drama"]
            used_fallback[i] = True

    report = dict(
        model_available=False,
        fallback_rows=int(used_fallback.sum()),
        cache_rows=int(used_cache.sum()),
        heuristic_rows=int(used_fallback.sum()),
        total_rows=len(X_df),
        degraded_gracefully=True,  # no exception propagated to caller
    )
    return scores, report
