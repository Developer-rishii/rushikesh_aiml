"""
engineer.py — Steps 2/3: derive candidate features from domain reasoning,
plus aggregate/ratio features. Every function here is documented with the
domain rationale (per the brief: "encoding subject knowledge into inputs")
and is computed row-wise from ALREADY-AVAILABLE measurements only — no
column here uses information that wouldn't exist at prediction time,
which is what keeps Step 5's leakage check trivial to pass rather than
something to route around.

WDBC background: each measurement (radius, texture, perimeter, area,
smoothness, compactness, concavity, concave points, symmetry, fractal
dimension) is recorded three ways per sample: mean, standard error, and
"worst" (largest/most extreme value observed across the cell nuclei in
the image). A pathologist reading these by eye looks not just at each
number alone but at how EXTREME the worst reading is relative to the
average, and how VARIABLE the readings are — that comparison is exactly
what raw per-column features don't encode on their own.
"""
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("src.features.engineer")

# The 10 base measurement names shared across mean_/_error/worst_ columns.
BASE_MEASUREMENTS = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave points", "symmetry", "fractal dimension",
]


def _col(prefix: str, measurement: str) -> str:
    return f"{prefix} {measurement}"


def derive_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain feature #1 — worst-to-mean ratio, per measurement:
      "how much more extreme is the worst cell than the average cell?"
      A tumor where the worst nucleus is 3x the average is read differently
      than one where worst ~ mean, even at the same mean value. This is
      literally what a pathologist eyeballs when scanning a slide.

    Domain feature #2 — error-to-mean ratio (coefficient of variation),
      per measurement: "how heterogeneous are the cells in this sample?"
      High cell-to-cell variability is itself a malignancy signal
      independent of the mean value.

    Both are computed strictly from columns that already exist for that
    same sample (mean_X, error_X, worst_X) — no future or target
    information enters either formula, by construction.
    """
    out = df.copy()
    added = []
    for m in BASE_MEASUREMENTS:
        mean_c, err_c, worst_c = _col("mean", m), _col(m, "error"), _col("worst", m)
        if mean_c in out.columns and worst_c in out.columns:
            ratio_col = f"ratio_worst_to_mean_{m.replace(' ', '_')}"
            out[ratio_col] = out[worst_c] / (out[mean_c].abs() + 1e-6)
            added.append(ratio_col)
        if mean_c in out.columns and err_c in out.columns:
            cv_col = f"coeff_variation_{m.replace(' ', '_')}"
            out[cv_col] = out[err_c] / (out[mean_c].abs() + 1e-6)
            added.append(cv_col)

    log.info("[Step 2] Derived %s domain ratio/variability features: %s", len(added), added[:4])
    return out, added


def derive_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain feature #3 — composite severity scores: the mean across
    related 'worst_*' shape-irregularity measurements (concavity, concave
    points, compactness) into one aggregate. Rationale: shape irregularity
    measurements are correlated by nature (they all describe how
    non-circular the nucleus boundary is) — a single aggregate is a
    common way domain experts summarize "how irregular is this cell
    overall" without needing every sub-measurement individually.
    """
    out = df.copy()
    added = []
    shape_cols = [c for c in ["worst concavity", "worst concave points", "worst compactness"]
                  if c in out.columns]
    if shape_cols:
        out["aggregate_worst_shape_irregularity"] = out[shape_cols].mean(axis=1)
        added.append("aggregate_worst_shape_irregularity")

    size_cols = [c for c in ["worst radius", "worst perimeter", "worst area"] if c in out.columns]
    if size_cols:
        z = (out[size_cols] - out[size_cols].mean()) / out[size_cols].std(ddof=0)
        out["aggregate_worst_size_zscore"] = z.mean(axis=1)
        added.append("aggregate_worst_size_zscore")

    log.info("[Step 3] Derived %s aggregate features: %s", len(added), added)
    return out, added
