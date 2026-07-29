"""
features.py
============
SINGLE SOURCE OF TRUTH for feature computation. Both train_ranker.py and
serve.py import FEATURE_COLUMNS and compute_features() from here. This is
the "disciplined feature-computation layer" the study guide calls for as
the fix to train/serve skew (Section 5, "the single biggest silent killer").

A hash of this file's feature list + logic is stamped into every registry
entry (see registry.py: feature_schema_hash) so that at serve time we can
detect if the deployed model expects a schema this code no longer produces.
"""
import hashlib
import inspect

FEATURE_COLUMNS = [
    "skill_match_score",
    "embedding_similarity",
    "experience_years",
    "location_match",
    "recruiter_response_rate",
    "past_ctr",
]

PROTECTED_ATTRIBUTE = "gender"  # used only for fairness audit, never as a model feature


def compute_features(df):
    """Return the exact feature matrix (as a DataFrame) the model consumes.

    Kept intentionally trivial (pass-through + dtype coercion) here because
    the raw log already contains clean numeric columns, but every real
    transformation (scaling, missing-value fill, clipping) MUST live in this
    one function so training and serving can never diverge.
    """
    out = df[FEATURE_COLUMNS].copy()
    out = out.fillna(out.median(numeric_only=True))
    for c in FEATURE_COLUMNS:
        out[c] = out[c].astype(float)
    return out


def feature_schema_hash() -> str:
    """Fingerprint of the feature contract (columns + this function's source).
    Changes if anyone edits feature logic without bumping a model version."""
    src = inspect.getsource(compute_features) + "|".join(FEATURE_COLUMNS)
    return hashlib.sha256(src.encode()).hexdigest()[:16]
