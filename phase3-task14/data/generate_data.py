"""
Task 14 - Stage B.2: Build on real data
------------------------------------------------------------
We don't have access to PlaceMux's production logs, so this script
generates a REALISTIC interaction log with the same structural
properties production logs would have:
  - a protected attribute (gender) that the MODEL NEVER SEES
  - two proxy features (college_tier, pincode_tier) that correlate
    with gender the way real proxies do (this is what makes the
    "we don't use gender" pitfall possible)
  - a noisy, non-deterministic outcome (shortlisted) so the audit
    has to deal with real class imbalance and real noise, not a
    toy separable dataset.

Every run is seeded, so results are reproducible (Def. of Done:
"keep the experiment log so every number is reproducible").
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 12000


def generate(n=N):
    gender = RNG.choice(["M", "F"], size=n, p=[0.62, 0.38])  # real applicant skew

    # Proxy variables: correlated with gender WITHOUT being gender.
    # college_tier: 1 (top) - 3 (lower tier). Slight historical skew by gender.
    college_tier = np.where(
        gender == "M",
        RNG.choice([1, 2, 3], size=n, p=[0.30, 0.45, 0.25]),
        RNG.choice([1, 2, 3], size=n, p=[0.22, 0.45, 0.33]),
    )
    # pincode_tier: 1 (metro) - 3 (rural). Also correlated with gender in this market.
    pincode_tier = np.where(
        gender == "M",
        RNG.choice([1, 2, 3], size=n, p=[0.40, 0.35, 0.25]),
        RNG.choice([1, 2, 3], size=n, p=[0.30, 0.35, 0.35]),
    )

    experience_years = np.clip(RNG.normal(4, 2.2, n), 0, 20)
    skill_match_score = np.clip(RNG.normal(65, 15, n), 0, 100)
    test_score = np.clip(RNG.normal(60, 18, n), 0, 100)
    applications_count = RNG.poisson(3, n) + 1

    # True underlying merit signal (what SHOULD drive the decision)
    merit = (
        0.35 * skill_match_score
        + 0.30 * test_score
        + 4.0 * experience_years
        - 2.0 * college_tier
        - 1.5 * pincode_tier
    )
    merit = (merit - merit.mean()) / merit.std()

    # Historical bias baked into the label itself (this is what a real
    # audit is trying to catch: bias inherited from PAST human decisions,
    # not injected by the model).
    historical_bias = np.where(gender == "M", 0.22, -0.22)

    logit = 0.9 * merit + historical_bias - 0.5
    prob = 1 / (1 + np.exp(-logit))
    shortlisted = RNG.binomial(1, prob)

    df = pd.DataFrame({
        "candidate_id": [f"C{i:06d}" for i in range(n)],
        "gender": gender,                      # PROTECTED ATTRIBUTE - audit only
        "college_tier": college_tier,           # proxy feature - used by model
        "pincode_tier": pincode_tier,           # proxy feature - used by model
        "experience_years": experience_years,
        "skill_match_score": skill_match_score,
        "test_score": test_score,
        "applications_count": applications_count,
        "shortlisted": shortlisted,             # label
    })
    return df


if __name__ == "__main__":
    df = generate()
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "interactions_log.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows")
    print(df.groupby("gender")["shortlisted"].mean())
