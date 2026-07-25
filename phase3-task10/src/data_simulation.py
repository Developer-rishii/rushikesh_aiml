"""
data_simulation.py
-------------------
We do not have access to Altrodav/PlaceMux's real production logs, so this
module builds a data generator that plays the same role real logged data
would: candidate x job-posting pairs, a graded relevance label, engagement
events (impression -> click -> apply -> shortlist) generated through a
realistic position-biased funnel, and a synthetic candidate "segment" used
purely as a fairness-audit proxy (NOT a real protected attribute — see
README for why a synthetic proxy is used instead of real demographic data).

This is documented explicitly as a stand-in for real logs (Stage A, item 2:
"confirm every prerequisite is genuinely in hand" — it is not, so we build
the closest honest substitute and say so, rather than pretending).

Everything downstream (training, evaluation, the A/B test, the readout) is
computed the same way it would be on real logs: nothing about the ranker,
the stats test, or the decision code depends on this being synthetic.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

N_QUERIES_HIST = 800          # job postings in the historical/offline window
CANDIDATES_PER_QUERY = 40     # candidates considered per posting
N_DAYS_HIST = 60


def _latent_relevance(skill_match, experience_years, distance_km, embedding_sim):
    """Ground-truth (unobservable in real life) relevance function.
    A real system never gets to see this directly — it only sees noisy
    engagement events generated from it. We keep it here only to drive the
    simulation, and the training/eval code never reads this function.
    """
    score = (
        2.2 * skill_match
        + 0.35 * np.log1p(experience_years)
        - 0.015 * distance_km
        + 1.6 * embedding_sim
    )
    return score


def _grade(score):
    """Convert a continuous latent score into a 0-3 graded relevance label,
    the standard target for learning-to-rank (nDCG/MAP expect graded labels).
    """
    q = np.quantile(score, [0.55, 0.80, 0.95])
    grade = np.digitize(score, q)
    return grade


def simulate_historical_logs(seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for qid in range(N_QUERIES_HIST):
        posted_day = int(rng.integers(0, N_DAYS_HIST))
        n_cand = CANDIDATES_PER_QUERY
        skill_match = np.clip(rng.beta(2.2, 2.0, n_cand), 0, 1)
        experience_years = np.clip(rng.gamma(2.0, 2.0, n_cand), 0, 25)
        distance_km = np.clip(rng.exponential(15, n_cand), 0, 200)
        past_ctr = np.clip(rng.beta(1.5, 8, n_cand), 0, 1)
        embedding_sim = np.clip(rng.normal(0.5, 0.18, n_cand), 0, 1)
        segment = rng.choice(["segment_A", "segment_B"], size=n_cand, p=[0.5, 0.5])

        latent = _latent_relevance(skill_match, experience_years, distance_km, embedding_sim)
        latent += rng.normal(0, 0.35, n_cand)  # observation noise
        grade = _grade(latent)

        for i in range(n_cand):
            rows.append(
                dict(
                    query_id=qid,
                    candidate_id=f"{qid}_{i}",
                    posted_day=posted_day,
                    skill_match=skill_match[i],
                    experience_years=experience_years[i],
                    distance_km=distance_km[i],
                    past_ctr=past_ctr[i],
                    embedding_sim=embedding_sim[i],
                    segment=segment[i],
                    relevance=int(grade[i]),
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Online A/B simulation
# ---------------------------------------------------------------------------
# The TRUE effect is fixed here, once, and is treated as a secret the
# readout code (readout.py) is never allowed to import or read. This is what
# makes it a fair test of the analysis code: readout.py has to recover the
# right answer purely from simulated events, the way a real analyst would
# from logs, with no privileged access to ground truth.

TRUE_TREATMENT_EFFECT_ON_RELEVANCE = 0.12   # small, realistic-sized true lift
N_QUERIES_ONLINE = 900          # ~ new postings shown over the 14-day test
CANDIDATES_SHOWN = 10           # top-k actually rendered to a recruiter
TEST_DURATION_DAYS = 14


def _position_bias(rank):
    # Classic 1/log2(rank+1)-style decay: top-of-list gets seen/clicked more.
    return 1.0 / np.log2(rank + 2)


def _rank_candidates(df_query, model_scores):
    order = np.argsort(-model_scores)
    return df_query.iloc[order].reset_index(drop=True)


def simulate_ab_events(baseline_score_fn, treatment_score_fn, seed=7):
    """Generate impression/click/application/shortlist events for CONTROL
    (baseline ranker) and TREATMENT (new ranker) arms over a fixed 14-day
    horizon, with realistic position bias and a small true underlying lift
    for treatment. Assignment is by query (job posting), 50/50.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for qid in range(N_QUERIES_ONLINE):
        day = int(rng.integers(0, TEST_DURATION_DAYS))
        n_cand = CANDIDATES_PER_QUERY
        skill_match = np.clip(rng.beta(2.2, 2.0, n_cand), 0, 1)
        experience_years = np.clip(rng.gamma(2.0, 2.0, n_cand), 0, 25)
        distance_km = np.clip(rng.exponential(15, n_cand), 0, 200)
        past_ctr = np.clip(rng.beta(1.5, 8, n_cand), 0, 1)
        embedding_sim = np.clip(rng.normal(0.5, 0.18, n_cand), 0, 1)
        segment = rng.choice(["segment_A", "segment_B"], size=n_cand, p=[0.5, 0.5])

        latent = _latent_relevance(skill_match, experience_years, distance_km, embedding_sim)
        arm = "treatment" if rng.random() < 0.5 else "control"
        true_latent = latent + (TRUE_TREATMENT_EFFECT_ON_RELEVANCE if arm == "treatment" else 0.0)
        true_latent_noisy = true_latent + rng.normal(0, 0.35, n_cand)

        cand_df = pd.DataFrame(
            dict(
                candidate_id=[f"on_{qid}_{i}" for i in range(n_cand)],
                skill_match=skill_match,
                experience_years=experience_years,
                distance_km=distance_km,
                past_ctr=past_ctr,
                embedding_sim=embedding_sim,
                segment=segment,
                posted_day=day,
            )
        )
        score_fn = treatment_score_fn if arm == "treatment" else baseline_score_fn
        model_scores = score_fn(cand_df, as_of_day=day)
        ranked = _rank_candidates(cand_df, model_scores)
        true_latent_sorted = true_latent_noisy[np.argsort(-model_scores)]

        shown = ranked.iloc[:CANDIDATES_SHOWN].reset_index(drop=True)
        true_shown = true_latent_sorted[:CANDIDATES_SHOWN]

        for rank, (_, cand) in enumerate(shown.iterrows()):
            pb = _position_bias(rank)
            p_click = np.clip(pb * (0.10 + 0.20 * (true_shown[rank] > np.median(true_latent))), 0, 1)
            clicked = rng.random() < p_click
            p_apply = np.clip(0.35 * pb if clicked else 0.0, 0, 1) * (1 + 0.5 * (true_shown[rank] > np.percentile(true_latent, 70)))
            applied = clicked and (rng.random() < min(p_apply, 1.0))
            p_shortlist = 0.28 if applied else 0.0
            shortlisted = applied and (rng.random() < p_shortlist)

            rows.append(
                dict(
                    query_id=qid,
                    day=day,
                    arm=arm,
                    candidate_id=cand["candidate_id"],
                    rank=rank,
                    segment=cand["segment"],
                    impression=1,
                    click=int(clicked),
                    application=int(applied),
                    shortlist=int(shortlisted),
                )
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = simulate_historical_logs()
    df.to_csv("data/historical_logs.csv", index=False)
    print(f"wrote data/historical_logs.csv rows={len(df)} queries={df.query_id.nunique()}")
