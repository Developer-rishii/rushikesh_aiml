from src.cold_start_recommender import ColdStartRecommender
from src.features import skill_overlap_vector

def explain_recommendation(user_skills, job_df, job_id, recommender: ColdStartRecommender):
    job_row = job_df[job_df.job_id == job_id].iloc[0]
    overlap = skill_overlap_vector(user_skills, job_df[job_df.job_id == job_id])[0]
    return {
        "input_user_skills": list(user_skills),
        "output_job_id": int(job_id),
        "output_job_skills": job_row["skills"],
        "reason": (f"Recommended because {round(overlap*100)}% skill overlap with your profile "
                   f"({sorted(set(user_skills) & set(job_row['skills']))} in common), "
                   f"weighted {recommender.w_overlap} toward relevance and "
                   f"{recommender.w_popularity} toward popularity, "
                   f"with {int(recommender.epsilon*100)}% exploration slots reserved to learn your taste faster.")
    }