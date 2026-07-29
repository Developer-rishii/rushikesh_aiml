"""
Feature contract for the model.

MODEL_FEATURES deliberately excludes `gender`. This is the exact
setup the pitfalls section warns about: "'We don't use gender' treated
as proof of fairness." college_tier and pincode_tier are legitimate
looking proxy features that correlate with gender in this data - the
whole point of the audit is to prove that omission alone doesn't buy
fairness.
"""
MODEL_FEATURES = [
    "experience_years",
    "skill_match_score",
    "test_score",
    "applications_count",
    "college_tier",
    "pincode_tier",
]

PROTECTED_ATTRIBUTE = "gender"
LABEL = "shortlisted"
