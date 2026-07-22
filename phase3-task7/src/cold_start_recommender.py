import numpy as np
from src.features import build_feature_matrix
from src.popularity import top_popular

MODEL_VERSION = "1.0.0"

class ColdStartRecommender:
    def __init__(self, w_overlap=0.7, w_popularity=0.3, epsilon=0.15):
        self.w_overlap = w_overlap
        self.w_popularity = w_popularity
        self.epsilon = epsilon  # exploration rate
        self.version = MODEL_VERSION

    def get_metadata(self):
        return {
            "model_version": self.version,
            "w_overlap": self.w_overlap,
            "w_popularity": self.w_popularity,
            "epsilon": self.epsilon,
            "algorithm": "Hybrid Content-Popularity Cold-Start with Tiered Exploration"
        }

    def score(self, user_skills, job_df):
        if len(job_df) == 0:
            return np.array([])
        feats = build_feature_matrix(user_skills, job_df)
        overlap, pop = feats[:, 0], feats[:, 1]
        # normalize popularity to comparable scale
        pop_range = (pop.max() - pop.min())
        pop_norm = (pop - pop.min()) / (pop_range + 1e-9) if pop_range > 0 else np.zeros_like(pop)
        return self.w_overlap * overlap + self.w_popularity * pop_norm

    def recommend(self, user_skills, job_df, k=10, rng=None, return_details=False):
        if len(job_df) == 0:
            return [] if not return_details else {"job_ids": [], "details": [], "version": self.version}
        rng = rng or np.random.default_rng()
        scores = self.score(user_skills, job_df)
        ranked_idx = np.argsort(-scores)

        n_explore = max(1, int(k * self.epsilon)) if len(user_skills) > 0 else k
        n_exploit = k - n_explore

        exploit_indices = ranked_idx[:n_exploit]
        exploit_ids = job_df.iloc[exploit_indices]["job_id"].tolist()

        # exploration: sample from the NEXT tier down (uncertain-but-plausible), not pure random —
        # keeps exploration from tanking first-session relevance.
        explore_pool = ranked_idx[n_exploit:n_exploit + 50]
        if len(explore_pool) > 0:
            chosen_explore = rng.choice(
                explore_pool,
                size=min(n_explore, len(explore_pool)),
                replace=False
            )
            explore_ids = job_df.iloc[chosen_explore]["job_id"].tolist()
        else:
            explore_ids = []

        recs = exploit_ids + explore_ids
        recs = recs[:k]

        if return_details:
            details = []
            for jid in recs:
                is_explore = jid in explore_ids
                details.append({
                    "job_id": int(jid),
                    "is_exploration": is_explore,
                    "slot_type": "explore" if is_explore else "exploit"
                })
            return {"job_ids": recs, "details": details, "version": self.version}

        return recs