"""
Stage D - Per-decision explanations.

Because the model is Logistic Regression, the contribution of each
feature to a single decision is EXACT (not an approximation like
SHAP-on-a-blackbox):

    logit = intercept + sum_i ( coef_i * scaled_feature_i )

We report each term's contribution to the logit, rank by |contribution|,
and template the top drivers into plain English a rejected candidate
(or a regulator) can actually read.
"""
import numpy as np
from features import MODEL_FEATURES

FRIENDLY_NAMES = {
    "experience_years": "years of experience",
    "skill_match_score": "skill match with the role",
    "test_score": "assessment test score",
    "applications_count": "number of applications submitted",
    "college_tier": "college tier on file",
    "pincode_tier": "location tier on file",
}

# Direction hint: for these, LOWER tier number is better (tier 1 = top),
# so a positive raw value with a negative coefficient still needs a
# human-readable direction, not just "went up".
LOWER_IS_BETTER = {"college_tier", "pincode_tier"}


def explain_decision(clf, scaler, row_features: dict, threshold=0.5, top_k=3):
    x = np.array([[row_features[f] for f in MODEL_FEATURES]])
    xs = scaler.transform(x)[0]

    contributions = clf.coef_[0] * xs
    intercept = clf.intercept_[0]
    logit = intercept + contributions.sum()
    proba = 1 / (1 + np.exp(-logit))
    decision = int(proba >= threshold)

    terms = []
    for feat, contrib, raw in zip(MODEL_FEATURES, contributions, x[0]):
        terms.append({
            "feature": feat,
            "friendly_name": FRIENDLY_NAMES[feat],
            "raw_value": float(raw),
            "contribution_to_logit": round(float(contrib), 4),
            "direction": "helped" if contrib > 0 else "hurt",
        })
    terms_sorted = sorted(terms, key=lambda t: -abs(t["contribution_to_logit"]))

    plain_english = _to_plain_english(terms_sorted[:top_k], decision, proba)

    return {
        "decision": "shortlisted" if decision else "not shortlisted",
        "probability": round(float(proba), 4),
        "threshold": threshold,
        "top_factors": terms_sorted[:top_k],
        "all_factors": terms_sorted,
        "explanation": plain_english,
    }


def _to_plain_english(top_terms, decision, proba):
    verdict = "was shortlisted" if decision else "was not shortlisted"
    lines = [f"This candidate {verdict} (model confidence: {proba:.0%})."]
    for t in top_terms:
        name = t["friendly_name"]
        helped = t["contribution_to_logit"] > 0
        if t["feature"] in LOWER_IS_BETTER:
            lines.append(
                f"- {name.capitalize()} (tier {int(t['raw_value'])}) "
                f"{'worked in their favor' if helped else 'worked against them'}."
            )
        else:
            lines.append(
                f"- {name.capitalize()} of {t['raw_value']:.1f} "
                f"{'worked in their favor' if helped else 'worked against them'}."
            )
    return " ".join(lines)


def model_unavailable_fallback():
    """
    Stage B.4 / D.4 requirement: "what happens when the model is
    unavailable". We never silently guess a decision - we return an
    explicit, auditable degraded response instead of a fabricated score.
    """
    return {
        "decision": "DEFERRED_TO_HUMAN_REVIEW",
        "probability": None,
        "explanation": (
            "The scoring model is currently unavailable. This application "
            "has been queued for manual review instead of being auto-scored "
            "or auto-rejected. No automated decision was made."
        ),
        "degraded_mode": True,
    }
