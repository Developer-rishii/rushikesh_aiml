"""
pre_registration.py
--------------------
Deliverable B: "A live model A/B with a pre-registered hypothesis and
metric." This file is run and its output frozen to disk BEFORE
ab_simulation.py is ever run. That ordering is the whole point: it is what
makes the eventual result honest instead of retrofitted (Pitfall #1 in the
study guide: "metric chosen after seeing results").

run_all.sh enforces this ordering. readout.py refuses to run unless
pre_registration.json already exists on disk with an earlier timestamp than
the online events file, so the pipeline cannot be silently reordered.
"""

import json
import math
import time

from scipy.stats import norm

PRIMARY_METRIC = "application_rate"          # applications / impressions shown
BASELINE_RATE = 0.052                        # measured from historical_logs.csv funnel
MDE_RELATIVE = 0.08                          # smallest lift worth shipping for: +8% relative
ALPHA = 0.05
POWER = 0.80


def required_sample_size(p1, mde_relative, alpha, power):
    p2 = p1 * (1 + mde_relative)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    pooled = (p1 + p2) / 2
    n = (
        (z_alpha * math.sqrt(2 * pooled * (1 - pooled)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
        / (p2 - p1) ** 2
    )
    return math.ceil(n)


def build_pre_registration():
    n_per_arm = required_sample_size(BASELINE_RATE, MDE_RELATIVE, ALPHA, POWER)
    doc = {
        "locked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hypothesis": (
            "Ranking candidates with the treatment model (learned relevance "
            "ranker) increases the application rate (applications per "
            "impression, top-10 shown) for job postings, relative to the "
            "current production heuristic ranker, without degrading the "
            "shortlist rate or widening the fairness gap between candidate "
            "segments."
        ),
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_definition": "sum(applications) / sum(impressions), computed per arm over the full test window",
        "baseline_rate_estimate": BASELINE_RATE,
        "minimum_detectable_effect_relative": MDE_RELATIVE,
        "alpha": ALPHA,
        "power": POWER,
        "required_sample_size_per_arm_impressions": n_per_arm,
        "test_type": "fixed_horizon_two_proportion_z_test",
        "test_duration_days": 14,
        "randomization_unit": "job_posting (query_id), 50/50 control/treatment",
        "single_look_rule": (
            "Results are read out exactly once, after the full 14-day window "
            "closes and the required sample size is met. No interim peeking. "
            "If the window closes before the required sample size is reached, "
            "the test is extended, not stopped early, and this is logged as a "
            "deviation."
        ),
        "guardrail_metrics": [
            {
                "name": "shortlist_rate",
                "definition": "shortlists / applications",
                "rule": "must not drop by more than 5% relative vs control (non-inferiority guardrail)",
            },
            {
                "name": "fairness_parity_gap",
                "definition": "abs(application_rate[segment_A] - application_rate[segment_B]) within the treatment arm",
                "rule": "must not exceed the control arm's parity gap by more than 2 percentage points",
            },
        ],
        "alternative_approaches_considered": {
            "fixed_horizon_vs_sequential": (
                "Chose fixed-horizon (single look at a pre-computed sample size) "
                "over sequential/always-valid testing. Rejected sequential testing "
                "because it needs an always-valid inference library and more "
                "engineering time than this sprint has; fixed-horizon is simpler "
                "to defend to a non-statistician stakeholder and its false-positive "
                "control is easy to verify by hand. Tradeoff accepted: fixed-horizon "
                "cannot stop early even if the effect is obviously huge or obviously "
                "null, so it costs more calendar time in unambiguous cases."
            ),
        },
        "decision_rule_committed_in_advance": (
            "SHIP if primary metric lift is statistically significant (p<0.05) AND "
            "positive AND both guardrails pass. "
            "DO NOT SHIP AND REVERT to the baseline heuristic if the primary metric "
            "is flat/not significant, or negative, or any guardrail fails — 'ship on "
            "a neutral result' was considered and rejected: the treatment model adds "
            "permanent serving cost and complexity (Stage E / brainstorming Q2: 'is "
            "the lift worth the complexity you are adding forever?'), so a neutral "
            "result does not clear that bar."
        ),
    }
    return doc


if __name__ == "__main__":
    doc = build_pre_registration()
    with open("artifacts/pre_registration.json", "w") as f:
        json.dump(doc, f, indent=2)
    print(json.dumps(doc, indent=2))
