"""
generate_data.py
-----------------
Produces a realistic candidate<->job interaction log for the PlaceMux
matching/ranking service.

NOTE ON DATA PROVENANCE (read this before you present the project):
We do not have access to Altrodav's real production database from this
environment. Rather than fabricate a claim of "real production logs", this
script generates a statistically realistic interaction log: impressions,
clicks, applications and shortlists, with the same schema, class imbalance,
and noise properties you'd expect from a live marketplace (heavy positional
bias, sparse positive labels, feature interactions that actually predict
outcome). Every downstream script (training, serving, load test) treats this
file exactly as it would treat a real log export -- same code path. Swap this
file for a real `SELECT * FROM impressions JOIN outcomes ...` export and
nothing else in the pipeline changes.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_JOBS = 400
CANDS_PER_JOB = 60          # impressions shown per job posting (ranked list length)
OUT_PATH = "data/interaction_logs.csv"


def make_job_features(n_jobs):
    return pd.DataFrame({
        "job_id": np.arange(n_jobs),
        "job_seniority": RNG.integers(0, 4, n_jobs),          # 0=intern..3=senior
        "job_urgency_score": RNG.uniform(0, 1, n_jobs),       # hiring urgency
        "job_num_applicants_so_far": RNG.integers(0, 200, n_jobs),
    })


def make_rows(jobs):
    rows = []
    for _, job in jobs.iterrows():
        n = CANDS_PER_JOB
        exp_years = RNG.gamma(2.0, 2.0, n).clip(0, 20)
        skill_match = RNG.beta(2, 2, n)              # 0..1 embedding-similarity proxy
        education_score = RNG.uniform(0, 1, n)
        location_match = RNG.integers(0, 2, n)
        past_response_rate = RNG.beta(2, 5, n)
        position_in_feed = RNG.permutation(n)         # 0 = top of ranked list shown

        # Ground-truth propensity: what actually drives a real outcome.
        # This is the "true" relationship the model has to recover from noisy clicks.
        true_affinity = (
            2.2 * skill_match
            + 0.9 * (exp_years / 20)
            + 0.5 * education_score
            + 0.6 * location_match
            + 0.8 * past_response_rate
            + 0.4 * (job["job_urgency_score"])
            - 0.35 * (position_in_feed / n)            # position bias: lower rank -> less seen
        )
        true_affinity = (true_affinity - true_affinity.mean()) / (true_affinity.std() + 1e-6)
        p_click = 1 / (1 + np.exp(-(true_affinity - 0.4)))
        click = RNG.binomial(1, p_click * 0.35)         # clicks are sparse

        p_apply_given_click = 1 / (1 + np.exp(-(true_affinity - 0.2)))
        applied = click * RNG.binomial(1, p_apply_given_click * 0.5)

        p_shortlist_given_apply = 1 / (1 + np.exp(-(true_affinity + 0.1)))
        shortlisted = applied * RNG.binomial(1, p_shortlist_given_apply * 0.4)

        # relevance label used for learning-to-rank (pointwise proxy)
        relevance = click + applied + shortlisted  # 0..3, monotonic w/ engagement depth

        for i in range(n):
            rows.append((
                job["job_id"], f"cand_{job['job_id']}_{i}",
                exp_years[i], skill_match[i], education_score[i],
                location_match[i], past_response_rate[i], position_in_feed[i],
                job["job_seniority"], job["job_urgency_score"], job["job_num_applicants_so_far"],
                int(click[i]), int(applied[i]), int(shortlisted[i]), int(relevance[i]),
            ))
    cols = ["job_id", "candidate_id", "exp_years", "skill_match", "education_score",
            "location_match", "past_response_rate", "position_in_feed",
            "job_seniority", "job_urgency_score", "job_num_applicants_so_far",
            "click", "applied", "shortlisted", "relevance"]
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    jobs = make_job_features(N_JOBS)
    df = make_rows(jobs)
    df.to_csv(OUT_PATH, index=False)
    print(f"wrote {len(df):,} rows across {df['job_id'].nunique()} jobs -> {OUT_PATH}")
    print(df[["click", "applied", "shortlisted", "relevance"]].mean())
