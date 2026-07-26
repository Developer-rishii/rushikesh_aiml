"""
serve.py
========
Stage B.4 / Stage E: "plus what happens when the model is unavailable."
This is the ONLY place that decides what a candidate ranking looks like
at request time. It tries the LTR model; if feature-building, the model
file, or scoring throws for ANY reason, it falls back to the heuristic
baseline and logs the fallback -- never a 500, never an empty ranking.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from features import build_features, FEATURE_COLUMNS
from heuristic_baseline import score as heuristic_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("serve")

MODEL_PATH = str(ROOT / "artifacts" / "w_pairwise_corrected.npy")


class Ranker:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._w = None

    def _load(self):
        if self._w is None:
            self._w = np.load(self.model_path)
        return self._w

    def rank(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Returns candidates sorted best-first, with a `served_by` column
        recording whether the LTR model or the heuristic fallback produced
        the order (this is what gets logged for the next round of LTR
        training and for the drift/monitoring dashboard)."""
        out = candidates.copy()
        try:
            w = self._load()
            feat = build_features(out)
            out["score"] = feat[FEATURE_COLUMNS].values @ w
            out["served_by"] = "ltr_pairwise_corrected"
        except Exception as e:
            log.warning(f"LTR scoring failed ({e!r}) -- falling back to heuristic baseline.")
            out["score"] = heuristic_score(out)
            out["served_by"] = "heuristic_fallback"
        return out.sort_values("score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "raw_logs.csv")
    sample = df[df.job_id == df.job_id.iloc[0]]
    ranker = Ranker()
    ranked = ranker.rank(sample)
    print(ranked[["candidate_idx", "score", "served_by"]].head(5).to_string(index=False))
