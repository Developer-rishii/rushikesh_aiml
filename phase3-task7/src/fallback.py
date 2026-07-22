from src.popularity import top_popular

EVERGREEN_JOB_IDS = [0, 1, 2, 3, 4]  # curated, always-open reqs — last-resort floor

def get_recommendations(recommender, user_skills, job_df, k=10, model_available=True):
    version = getattr(recommender, "version", "1.0.0")
    try:
        if not model_available:
            raise RuntimeError("model unavailable")
        recs = recommender.recommend(user_skills, job_df, k=k)
        if recs:
            return {
                "source": "model",
                "job_ids": recs,
                "reason": "personalized skill+popularity blend",
                "model_version": version
            }
        raise ValueError("model returned empty set")
    except Exception as e:
        pop = top_popular(job_df, k=k)
        if pop:
            return {
                "source": "popularity_fallback",
                "job_ids": pop,
                "reason": f"model unavailable/empty ({str(e)}) — served top popular jobs",
                "model_version": version
            }
        return {
            "source": "evergreen_fallback",
            "job_ids": EVERGREEN_JOB_IDS[:k],
            "reason": "no job pool available — served evergreen listings",
            "model_version": version
        }