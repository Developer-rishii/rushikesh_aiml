"""Feature computation used identically at TRAIN and SERVE time (train/serve
skew is the #1 pitfall called out in the study guide -- solved here by having
exactly one function both paths import)."""

SKILLS = ["python", "sql", "ml", "react", "java", "aws", "communication", "leadership"]

FEATURE_COLUMNS = (
    [f"skill_{s}" for s in SKILLS]
    + [f"req_{s}" for s in SKILLS]
    + ["years_exp", "min_exp"]
)


def compute_features(df):
    """Returns the exact feature matrix used for both training and serving.
    Keeping this as the single source of truth is how we detect/avoid
    train/serve skew instead of hoping two copies of feature code stay in sync.
    """
    return df[FEATURE_COLUMNS].copy()
