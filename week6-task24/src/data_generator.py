"""
Synthetic PlaceMux candidate x job-posting scoring data, built to be
"real-shaped, even if small" per the study guide's prerequisite, but here
scaled up and deliberately messy so the pipeline has to survive real-data
conditions, not a happy-path toy set.

Design intent (carried over, this is the actual fix from the earlier failed
attempt at this task):
- `skill_score`, `years_exp`, `jd_match`, `portfolio_score` are generated
  INDEPENDENTLY of `college_tier`. Actual ability does not depend on which
  tier college a candidate attended.
- The *historical* recommendation label (what past recruiters actually did)
  DOES carry a prestige bonus/penalty by tier -- this is what makes the
  historical training data biased, a realistic proxy-bias scenario.
- A separate `fair_recommended` label (merit-only, no prestige term) is
  generated purely as a fairness-ceiling reference in audit.py -- never used
  for training.
- Realistic messiness is injected on top: missing values, duplicate rows,
  and outlier scores, at rates typical of production marketplace data.
"""
import numpy as np
import pandas as pd

PRESTIGE_BONUS = {1: 25.0, 2: 0.0, 3: -25.0}  # tier 1 = most "prestigious"


def generate(n: int = 40_000, n_jobs: int = 50, seed: int = 42,
             missing_rate: float = 0.03, duplicate_rate: float = 0.01,
             outlier_rate: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    job_id = rng.integers(0, n_jobs, size=n)
    job_seniority_weight = 0.5 + (job_id % 5) * 0.15  # jobs vary in how much they weight experience

    college_tier = rng.choice([1, 2, 3], size=n, p=[0.30, 0.40, 0.30])
    skill_score = np.clip(rng.normal(65, 15, n), 0, 100)
    years_exp = np.clip(rng.normal(4, 2, n), 0, 15)
    jd_match = np.clip(rng.normal(60, 20, n), 0, 100)
    portfolio_score = np.clip(rng.normal(55, 18, n), 0, 100)

    # Ground-truth merit, deliberately with NO tier term. Job-specific weighting
    # of experience makes this a multi-job marketplace, not one static rule.
    merit = (0.4 * skill_score + 0.25 * jd_match + 0.15 * portfolio_score
              + job_seniority_weight * years_exp)
    merit_noisy = merit + rng.normal(0, 4, n)
    fair_threshold = np.percentile(merit_noisy, 70)
    fair_recommended = (merit_noisy > fair_threshold).astype(int)

    # Historical (biased) label: same merit signal + a prestige term.
    prestige = np.vectorize(PRESTIGE_BONUS.get)(college_tier)
    biased_score = merit + prestige + rng.normal(0, 5, n)
    biased_threshold = np.percentile(biased_score, 70)
    historical_recommended = (biased_score > biased_threshold).astype(int)

    df = pd.DataFrame({
        "candidate_id": np.arange(n),
        "job_id": job_id,
        "college_tier": college_tier,
        "skill_score": skill_score,
        "years_exp": years_exp,
        "jd_match": jd_match,
        "portfolio_score": portfolio_score,
        "historical_recommended": historical_recommended,
        "fair_recommended": fair_recommended,
    })

    # --- realistic messiness, injected on top of the clean signal ---
    rng2 = np.random.default_rng(seed + 1)

    # 1. Missing values in a couple of feature columns (sensor/self-report gaps)
    for col in ["jd_match", "portfolio_score"]:
        mask = rng2.random(n) < missing_rate
        df.loc[mask, col] = np.nan

    # 2. Outliers: a small fraction of scores get corrupted to extreme values
    outlier_mask = rng2.random(n) < outlier_rate
    df.loc[outlier_mask, "years_exp"] = rng2.uniform(20, 40, outlier_mask.sum())

    # 3. Duplicate rows (re-submitted applications) -- common in marketplace logs
    n_dupes = int(n * duplicate_rate)
    if n_dupes > 0:
        dupe_idx = rng2.choice(df.index, size=n_dupes, replace=True)
        df = pd.concat([df, df.loc[dupe_idx]], ignore_index=True)

    return df


if __name__ == "__main__":
    from config import CANDIDATES_PATH
    df = generate()
    df.to_csv(CANDIDATES_PATH, index=False)
    print(f"rows={len(df)}  duplicated={df.duplicated().sum()}  "
          f"missing_jd_match={df['jd_match'].isna().sum()}  "
          f"missing_portfolio={df['portfolio_score'].isna().sum()}")
    print(df.groupby("college_tier")["skill_score"].mean())
    print(df.groupby("college_tier")["historical_recommended"].mean())
