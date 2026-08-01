"""
Explanation contract (Stage B step 4 + core concept #3).

WE PROMISE:
  - a 0-100 band score (not the raw model float)
  - a short list of plain-English reason strings ranked by contribution direction
  - the model version that produced the decision

WE NEVER EXPOSE:
  - raw feature weights / coefficients
  - the exact numeric feature vector used
  - enough precision in the score to let repeated queries reconstruct the
    decision boundary (see design decision: bucketed bands, in DESIGN_DECISIONS.md)
"""

READABLE_FEATURES = {
    "skill_overlap": "shared skills with the role",
    "seniority_gap": "seniority match with the role",
    "same_location": "location match",
    "recency_days": "how fresh the posting is",
    "candidate_activity": "candidate's recent activity level",
}

# direction: for seniority_gap and recency_days, LOWER is better; flip the sign
# when deciding whether a feature counted "for" or "against" the match.
LOWER_IS_BETTER = {"seniority_gap", "recency_days"}


def band_score(raw_score, lo=-2.0, hi=5.5):
    """Bucket a raw continuous score into a coarse 0-100 band. Coarse buckets are
    a deliberate anti-extraction measure: an attacker gets far less signal per
    query than from a raw float, which slows model-cloning attacks."""
    pct = (raw_score - lo) / (hi - lo)
    pct = max(0.0, min(1.0, pct))
    band = round(pct * 20) * 5  # snap to nearest 5, i.e. 21 possible values
    return int(band)


def build_explanation(feature_row: dict, importances: dict, top_n=3):
    """Turn internal features + importances into partner-safe plain English.
    Never returns the raw importances dict or the raw feature_row values."""
    contributions = []
    for feat, weight in importances.items():
        value = feature_row.get(feat, 0)
        favorable = value > 0
        if feat in LOWER_IS_BETTER:
            favorable = value <= 1  # small gap/recent posting reads as favorable
        contributions.append((abs(weight), READABLE_FEATURES.get(feat, feat), favorable))

    contributions.sort(key=lambda x: -x[0])
    reasons = []
    for _, label, favorable in contributions[:top_n]:
        reasons.append(f"{'Strong' if favorable else 'Weak'} {label}")
    return reasons


def unavailable_explanation():
    return ["Live scoring temporarily unavailable — showing rule-based fallback ranking."]
