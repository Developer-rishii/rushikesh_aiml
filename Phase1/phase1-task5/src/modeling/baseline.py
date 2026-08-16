"""
modeling/baseline.py — Step 1: an explicit, deliberately dumb baseline.
Every model in this project is judged against this, never against
"looks good" in isolation (the brief's #1 pitfall: "No baseline").
"""
from sklearn.dummy import DummyClassifier


def build_baseline(cfg):
    """Majority-class classifier: predicts the most frequent training label,
    every time, regardless of input. If a 'real' model can't beat this, it
    has learned nothing."""
    return DummyClassifier(strategy=cfg.baseline_strategy, random_state=cfg.seed)
