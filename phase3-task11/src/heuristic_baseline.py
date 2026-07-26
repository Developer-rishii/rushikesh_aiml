"""heuristic_baseline.py -- the CURRENT production ranker (hand-tuned
weights, never re-fit against outcomes). This is the bar the LTR model
must beat on offline metrics before it's allowed anywhere near an online
test. Weights intentionally mirror data/generate_logs.py's _heuristic_score
so the "baseline" is realistic: over-indexes on recency/profile
completeness, ignores past_response_rate entirely -- exactly the kind of
heuristic drift that accumulates in real systems over time."""
import pandas as pd

WEIGHTS = {
    "skill_match": 0.20,
    "experience_match": 0.15,
    "embedding_sim": 0.15,
    "recency": 0.30,
    "profile_completeness": 0.20,
}


def score(df: pd.DataFrame) -> pd.Series:
    """Weighted sum over WHATEVER of the heuristic's inputs are actually
    present, renormalized -- a fallback that can itself be taken down by
    a missing column is not a real fallback (see tests/test_failure_and_bias.py)."""
    available = {c: w for c, w in WEIGHTS.items() if c in df.columns}
    if not available:
        return pd.Series(0.0, index=df.index)  # last-resort neutral order
    total_w = sum(available.values())
    return sum(df[col] * (w / total_w) for col, w in available.items())
