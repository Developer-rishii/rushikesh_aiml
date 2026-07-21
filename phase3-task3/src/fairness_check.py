"""
Lightweight parity check: are average predicted scores wildly different
across the synthetic protected group, controlling for the fact that
`relevance` itself was NOT generated as a function of group (see
data/generate_data.py)? Run at BOTH baseline and optimized stages -- not
just once at the end -- per the pitfall "a fairness audit done once, at the
end, as a formality". This is a sanity check, not a full audit (out of
scope for a latency task), and the report says so explicitly.
"""
import numpy as np
import pandas as pd
from src.config import PROTECTED_GROUP_COL, LABEL_COL


def parity_check(df: pd.DataFrame, score_col: str, threshold=0.15):
    g = df.groupby(PROTECTED_GROUP_COL)[score_col].mean()
    gap = float(g.max() - g.min())
    rel_gap = gap / (abs(g.mean()) + 1e-9)
    flagged = rel_gap > threshold
    return {
        "group_means": g.to_dict(),
        "abs_gap": gap,
        "relative_gap": round(rel_gap, 4),
        "threshold": threshold,
        "flagged": bool(flagged),
    }
