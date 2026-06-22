from src.features.feature_engineering import extract_features

class ExplainerEngine:
    def __init__(self):
        pass
        
    def generate_explanation(self, student, job, match_score=None):
        """
        Generate human-readable explanations for candidate-job matches.
        """
        feats = extract_features(student, job)
        
        num_required = len(job.get("required_skills", "").split(",")) if job.get("required_skills") else 0
        
        # Qualification score status
        v_score = student.get("verified_score", 0)
        m_score = job.get("minimum_score", 0)
        
        if v_score >= m_score:
            score_status = f"exceeds the minimum qualification score ({v_score} >= {m_score})"
        else:
            score_status = f"falls short of the minimum qualification score ({v_score} < {m_score})"
            
        # Experience contribution
        exp_years = student.get("experience_years", 0)
        exp_text = f"They have {exp_years} years of experience."
        
        # Determine why matched
        if match_score is None:
            match_score = feats["skill_overlap_percentage"]
            
        is_matched = match_score > 65 and v_score >= (m_score - 5)
        
        if is_matched:
            decision_text = "The candidate is a strong match because they satisfy"
        else:
            decision_text = "The candidate is not a strong match. They satisfy"
            
        reason = (
            f"{decision_text} {feats['number_of_matching_skills']} out of {num_required} required skills "
            f"and {score_status}. {exp_text}"
        )
        
        return {
            "match_score": round(match_score, 2),
            "matched_skills": feats["matched_skills_list"],
            "missing_skills": feats["missing_skills_list"],
            "reason": reason
        }
