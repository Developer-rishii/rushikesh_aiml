"""
baseline_model.py

"Before any clever model, build a dumb baseline ... every later number is
only meaningful relative to this baseline." (Study guide, Section 4)

The baseline PlaceMux would ship on day one: rank/classify purely by
skill_overlap_score against a fixed threshold. No training, no other
signals. This gives every later metric (precision/recall/FPR of the real
ML model, and later of live production) a fixed point of comparison.
"""

import numpy as np

from src.config import TARGET_COLUMN


class SkillOverlapBaseline:
    """Predicts a match as 'successful' iff skill_overlap_score >= threshold."""

    def __init__(self, threshold=0.72):
        self.threshold = threshold

    def fit(self, df):
        """Pick the threshold that maximizes F1 on the given labeled data,
        so the baseline is the *best simple rule*, not a strawman."""
        best_f1, best_t = -1, self.threshold
        y = df[TARGET_COLUMN].values
        scores = df["skill_overlap_score"].values
        for t in np.linspace(0.4, 0.95, 56):
            pred = (scores >= t).astype(int)
            tp = ((pred == 1) & (y == 1)).sum()
            fp = ((pred == 1) & (y == 0)).sum()
            fn = ((pred == 0) & (y == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        self.threshold = float(best_t)
        return self

    def predict(self, df):
        return (df["skill_overlap_score"].values >= self.threshold).astype(int)

    def predict_proba(self, df):
        # Not a real probability - it's a rule. Expose 0/1 as the "score" so
        # it plugs into the same metrics code as the ML model.
        return self.predict(df).astype(float)
