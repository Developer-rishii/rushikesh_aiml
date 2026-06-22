from src.features.feature_engineering import extract_features

class RuleBasedMatcher:
    def __init__(self):
        pass
        
    def predict(self, student, job):
        """
        Baseline Rule-Based Matcher
        match_score = (matching_skills / required_skills) * 100
        """
        feats = extract_features(student, job)
        
        # Rule-based calculation from requirements
        match_score = feats["skill_overlap_percentage"]
        
        return {
            "match_score": round(match_score, 2),
            "matched_skills": feats["matched_skills_list"],
            "missing_skills": feats["missing_skills_list"]
        }
