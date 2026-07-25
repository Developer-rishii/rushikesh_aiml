"""
decision.py
-----------
Deliverable D: "A ship / do-not-ship decision with reasoning."

The rule applied here is copied verbatim from
pre_registration.json["decision_rule_committed_in_advance"] — this script
does not invent new logic after seeing the numbers. That is the entire
discipline this task is testing (study guide framing: "the discipline to
kill your own model if it lost").
"""

import json


def decide():
    with open("artifacts/pre_registration.json") as f:
        pre_reg = json.load(f)
    with open("artifacts/experiment_readout.json") as f:
        readout = json.load(f)

    primary = readout["primary_metric"]
    guardrails = readout["guardrails"]
    mde_relative = pre_reg["minimum_detectable_effect_relative"]

    significant = primary["significant_at_alpha_0.05"]
    positive = primary["absolute_diff"] > 0
    meets_mde = (primary["relative_lift_pct"] / 100) >= mde_relative
    guardrails_pass = guardrails["shortlist_rate"]["pass"] and guardrails["fairness_parity_gap"]["pass"]
    sample_ok = readout["sample_size"]["requirement_met"]

    reasons = []
    if not sample_ok:
        reasons.append("Required sample size per the pre-registered power analysis was not met — "
                        "result is inconclusive, not a null result.")
    if not significant:
        reasons.append(f"Primary metric (application_rate) lift of {primary['relative_lift_pct']:.2f}% "
                        f"is not statistically significant (p={primary['p_value']:.4f} >= 0.05).")
    if significant and not positive:
        reasons.append("Primary metric moved significantly in the WRONG direction.")
    if significant and positive and not meets_mde:
        reasons.append(f"Lift of {primary['relative_lift_pct']:.2f}% is statistically significant but "
                        f"below the pre-registered minimum detectable effect of {mde_relative*100:.1f}% — "
                        "real, but not judged worth the added model-serving complexity.")
    if not guardrails["shortlist_rate"]["pass"]:
        reasons.append(f"Shortlist-rate guardrail FAILED: relative change "
                        f"{guardrails['shortlist_rate']['relative_change']*100:.2f}% "
                        "(threshold: no more than -5%). This is the 'wins on clicks, loses on "
                        "shortlists' failure mode called out explicitly in the brainstorming section.")
    if not guardrails["fairness_parity_gap"]["pass"]:
        reasons.append("Fairness parity-gap guardrail FAILED: treatment widened the gap between "
                        "candidate segments beyond the pre-registered threshold.")

    ship = sample_ok and significant and positive and meets_mde and guardrails_pass

    if ship:
        headline = "SHIP"
        reasons = [f"Primary metric lift of {primary['relative_lift_pct']:.2f}% is statistically "
                   f"significant (p={primary['p_value']:.4f}), meets the pre-registered MDE "
                   f"({mde_relative*100:.1f}%), and both guardrails pass."] + reasons
    else:
        headline = "DO NOT SHIP — revert to baseline heuristic ranker"

    decision = {
        "decision": headline,
        "decision_rule_applied": pre_reg["decision_rule_committed_in_advance"],
        "reasoning": reasons,
        "inputs": {
            "primary_metric_relative_lift_pct": primary["relative_lift_pct"],
            "p_value": primary["p_value"],
            "sample_size_requirement_met": sample_ok,
            "shortlist_guardrail_pass": guardrails["shortlist_rate"]["pass"],
            "fairness_guardrail_pass": guardrails["fairness_parity_gap"]["pass"],
        },
        "rollback_plan": (
            "If DO NOT SHIP: traffic already defaults to control at the router level "
            "(see failure_test.py) — no rollback deploy needed, treatment is simply "
            "never promoted to 100%. Model version stays in the registry for postmortem."
        ),
    }

    with open("artifacts/ship_decision.json", "w") as f:
        json.dump(decision, f, indent=2)
    return decision


if __name__ == "__main__":
    decision = decide()
    print(json.dumps(decision, indent=2))
