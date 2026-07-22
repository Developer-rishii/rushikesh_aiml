import numpy as np
import pandas as pd
from src.cold_start_recommender import ColdStartRecommender, MODEL_VERSION
from src.popularity import top_popular
from data.generate_synthetic_data import skill_overlap
from src.skew_check import check_skew
from src.fairness import audit_fairness

def precision_at_k(relevant, ranked, k):
    ranked_k = ranked[:k]
    hits = sum(1 for j in ranked_k if j in relevant)
    return hits / k

def average_precision(relevant, ranked):
    hits, score = 0, 0.0
    for i, j in enumerate(ranked, start=1):
        if j in relevant:
            hits += 1
            score += hits / i
    return score / max(1, len(relevant))

def ndcg_at_k(relevant, ranked, k):
    def dcg(lst):
        return sum((1 if j in relevant else 0) / np.log2(i + 2) for i, j in enumerate(lst[:k]))
    ideal = dcg(list(relevant) + [j for j in ranked if j not in relevant])
    actual = dcg(ranked)
    return actual / ideal if ideal > 0 else 0.0

def simulate_first_session_relevance(user_skills, job_df, ranked_ids, threshold=0.34):
    """Ground truth for 'relevant' in a first session, since we have no click log yet
    for cold-start users by definition: skill-overlap above threshold = relevant.
    This mirrors how you'd label offline eval before online data exists."""
    overlap = {jid: skill_overlap(user_skills, job_df.loc[job_df.job_id == jid, "skills"].iloc[0])
               for jid in ranked_ids}
    return {jid for jid, ov in overlap.items() if ov >= threshold}

def run_offline_eval(users_cold, job_df, k=10, test_frac=0.3, seed=7):
    rng = np.random.default_rng(seed)
    n_test = int(len(users_cold) * test_frac)
    test_users = users_cold.sample(n=n_test, random_state=seed)  # held-out, not tuned on

    model = ColdStartRecommender()
    results = {"model": [], "popularity_only": []}

    for _, u in test_users.iterrows():
        model_ranked = model.recommend(u["skills"], job_df, k=k, rng=rng)
        pop_ranked = top_popular(job_df, k=k)

        relevant = simulate_first_session_relevance(u["skills"], job_df, model_ranked) | \
                   simulate_first_session_relevance(u["skills"], job_df, pop_ranked)

        results["model"].append({
            "p@5": precision_at_k(relevant, model_ranked, 5),
            "map": average_precision(relevant, model_ranked),
            "ndcg@10": ndcg_at_k(relevant, model_ranked, k),
        })
        results["popularity_only"].append({
            "p@5": precision_at_k(relevant, pop_ranked, 5),
            "map": average_precision(relevant, pop_ranked),
            "ndcg@10": ndcg_at_k(relevant, pop_ranked, k),
        })

    summary = {name: pd.DataFrame(rows).mean().to_dict() for name, rows in results.items()}
    return summary, n_test

def offline_vs_online_gap(summary):
    """Expected online effect is always a fraction of offline lift (novelty/position bias,
    train/serve skew). We report a conservative discount factor rather than assuming 1:1 transfer —
    this is the 'connect offline to online, treat online as truth' requirement."""
    offline_lift = summary["model"]["p@5"] - summary["popularity_only"]["p@5"]
    EXPECTED_ONLINE_DISCOUNT = 0.5  # conservative; tune from real A/B once logged
    expected_online_lift = offline_lift * EXPECTED_ONLINE_DISCOUNT
    return {
        "offline_p@5_lift": float(offline_lift),
        "expected_online_p@5_lift": float(expected_online_lift),
        "discount_applied": EXPECTED_ONLINE_DISCOUNT
    }

if __name__ == "__main__":
    jobs = pd.read_json("data/jobs.json")
    users = pd.read_json("data/users.json")
    cold_users = users[users["is_cold_start"]]

    print(f"=== Cold-Start Recommendation Model Evaluation (Version {MODEL_VERSION}) ===")

    # 1. Feature Train/Serve Skew Check
    sample_skills = cold_users["skills"].head(10).tolist()
    skew_res = check_skew(sample_skills, jobs)
    print(f"Train/Serve Skew Audit: max_diff = {skew_res['max_diff']} ({skew_res['status']})")

    # 2. Offline Model Evaluation vs Popularity Baseline
    summary, n_test = run_offline_eval(cold_users, jobs)
    print(f"Held-out cold-start test users: {n_test}")
    for name, metrics in summary.items():
        print(f"  - {name}: { {k: round(v, 4) for k, v in metrics.items()} }")

    gap = offline_vs_online_gap(summary)
    print(f"Offline p@5 lift over baseline: {round(gap['offline_p@5_lift'], 4)}")
    print(f"Expected online p@5 lift (discounted 50%): {round(gap['expected_online_p@5_lift'], 4)}")

    # 3. Fairness & Candidate Representation Audit
    recommender = ColdStartRecommender()
    fairness_res = audit_fairness(cold_users, jobs, recommender)
    print(f"Fairness Audit (DPDP/EEOC 80% Rule): Parity Ratio = {fairness_res['demographic_parity_ratio']} ({fairness_res['status']})")

    print("==========================================================================")