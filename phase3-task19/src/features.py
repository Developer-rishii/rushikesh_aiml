"""
Single feature-computation module, imported by BOTH training (model.py)
and serving (serve.py). This is the guardrail against train/serve skew
called out in the study guide as "the single biggest silent killer."

Never compute a feature inline in serve.py or model.py — always call
these functions from both places.
"""
FEATURE_COLUMNS = ["skill_overlap", "years_exp", "req_years_exp",
                    "distance_km", "num_skills", "exp_gap"]


def compute_features(df):
    """df must have: skill_overlap, years_exp, req_years_exp, distance_km,
    num_skills. Adds derived features. Returns df with FEATURE_COLUMNS
    present. Pure function -> identical result whether called during
    offline training or online serving."""
    df = df.copy()
    df["exp_gap"] = df["years_exp"] - df["req_years_exp"]
    return df


def assert_no_protected_attrs(columns):
    """Fails loudly if a protected-attribute proxy is ever passed into
    the feature set. Called by model.py before every training run."""
    banned = {"gender_proxy", "gender", "race", "religion", "caste",
              "age", "marital_status", "pregnancy", "disability"}
    hit = banned & set(columns)
    if hit:
        raise ValueError(f"Protected attribute(s) leaked into features: {hit}")
