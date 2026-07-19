"""
The study guide is explicit: "Before any clever model, build a dumb
baseline (e.g. rank by overlap of required vs verified skills)... every
later number is only meaningful relative to this baseline."

This is that baseline: a zero-training, rule-based recommender that simply
thresholds a weighted overlap of skill_score and jd_match. No fitting, no
tier awareness -- just the rule a recruiter might apply by eye.
"""
import numpy as np
import pandas as pd


def baseline_predict(df: pd.DataFrame, overlap_threshold: float = 60.0) -> np.ndarray:
    """Recommend if the simple overlap of skill and JD match clears a fixed
    bar. This is deliberately naive -- no learning, no per-job tuning."""
    overlap = 0.6 * df["skill_score"] + 0.4 * df["jd_match"]
    return (overlap >= overlap_threshold).astype(int).values


if __name__ == "__main__":
    from config import CANDIDATES_PATH
    from data_validation import clean
    df = clean(pd.read_csv(CANDIDATES_PATH))
    preds = baseline_predict(df)
    print(f"Baseline positive rate: {preds.mean():.3f}")
