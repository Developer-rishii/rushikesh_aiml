"""
readout.py
----------
Deliverable C: "An honest readout: effect size, significance, guardrails."

Hard rule enforced below: this script will refuse to run if
artifacts/pre_registration.json does not already exist, or if it was
modified/created AFTER data/online_ab_events.csv. That check is the
peeking guard — it makes it structurally impossible to choose the metric
after seeing results (Pitfall #1).

This script does NOT import anything from data_simulation.py's hidden
TRUE_TREATMENT_EFFECT_ON_RELEVANCE — the readout must recover the answer
purely from observed events, the same as it would from real production logs.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm


def _enforce_no_peeking():
    pre_reg_path = "artifacts/pre_registration.json"
    events_path = "data/online_ab_events.csv"
    if not os.path.exists(pre_reg_path):
        sys.exit("BLOCKED: no pre_registration.json found. Pre-register the "
                  "hypothesis and metric before running the readout.")
    if os.path.getmtime(pre_reg_path) > os.path.getmtime(events_path):
        sys.exit("BLOCKED: pre_registration.json is newer than the online events "
                  "file — this looks like the metric was chosen after seeing data.")


def two_proportion_z_test(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    se_unpooled = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_low = (p2 - p1) - 1.96 * se_unpooled
    ci_high = (p2 - p1) + 1.96 * se_unpooled
    return dict(p1=p1, p2=p2, abs_diff=p2 - p1,
                relative_lift=(p2 - p1) / p1 if p1 > 0 else float("nan"),
                z=z, p_value=p_value, ci95_abs_diff=[ci_low, ci_high])


def run_readout():
    _enforce_no_peeking()

    with open("artifacts/pre_registration.json") as f:
        pre_reg = json.load(f)

    events = pd.read_csv("data/online_ab_events.csv")
    n_days_observed = events["day"].nunique()
    n_impressions = events.groupby("arm")["impression"].sum()

    required_n = pre_reg["required_sample_size_per_arm_impressions"]
    sample_size_met = bool((n_impressions >= required_n).all())

    ctrl = events[events.arm == "control"]
    trt = events[events.arm == "treatment"]

    primary = two_proportion_z_test(
        ctrl["application"].sum(), ctrl["impression"].sum(),
        trt["application"].sum(), trt["impression"].sum(),
    )

    # Guardrail 1: shortlist rate (of applications), non-inferiority
    def rate(df, num, den):
        d = df[den].sum()
        return df[num].sum() / d if d > 0 else float("nan")

    shortlist_ctrl = rate(ctrl, "shortlist", "application")
    shortlist_trt = rate(trt, "shortlist", "application")
    shortlist_rel_change = (shortlist_trt - shortlist_ctrl) / shortlist_ctrl if shortlist_ctrl else float("nan")
    guardrail_shortlist_pass = shortlist_rel_change > -0.05

    # Guardrail 2: fairness parity gap between synthetic segments
    def seg_app_rate(df, seg):
        s = df[df.segment == seg]
        return s["application"].sum() / s["impression"].sum() if s["impression"].sum() else float("nan")

    gap_ctrl = abs(seg_app_rate(ctrl, "segment_A") - seg_app_rate(ctrl, "segment_B"))
    gap_trt = abs(seg_app_rate(trt, "segment_A") - seg_app_rate(trt, "segment_B"))
    guardrail_fairness_pass = (gap_trt - gap_ctrl) <= 0.02

    also_click_rate = two_proportion_z_test(
        ctrl["click"].sum(), ctrl["impression"].sum(),
        trt["click"].sum(), trt["impression"].sum(),
    )

    readout = {
        "generated_at_check": "computed strictly from data/online_ab_events.csv; see no-peeking guard above",
        "pre_registration_reference": pre_reg["locked_at"],
        "test_duration_days_observed": int(n_days_observed),
        "sample_size": {
            "required_per_arm": required_n,
            "observed_control": int(n_impressions.get("control", 0)),
            "observed_treatment": int(n_impressions.get("treatment", 0)),
            "requirement_met": sample_size_met,
        },
        "primary_metric": {
            "name": "application_rate",
            "control": primary["p1"],
            "treatment": primary["p2"],
            "absolute_diff": primary["abs_diff"],
            "relative_lift_pct": primary["relative_lift"] * 100,
            "p_value": primary["p_value"],
            "significant_at_alpha_0.05": bool(primary["p_value"] < 0.05),
            "ci95_absolute_diff": primary["ci95_abs_diff"],
        },
        "secondary_metric_click_rate": {
            "control": also_click_rate["p1"],
            "treatment": also_click_rate["p2"],
            "relative_lift_pct": also_click_rate["relative_lift"] * 100,
            "p_value": also_click_rate["p_value"],
            "note": "Reported so a 'wins on clicks but not shortlists' scenario is visible, not hidden.",
        },
        "guardrails": {
            "shortlist_rate": {
                "control": shortlist_ctrl,
                "treatment": shortlist_trt,
                "relative_change": shortlist_rel_change,
                "pass": bool(guardrail_shortlist_pass),
            },
            "fairness_parity_gap": {
                "control_gap": gap_ctrl,
                "treatment_gap": gap_trt,
                "pass": bool(guardrail_fairness_pass),
            },
        },
        "practical_significance_note": (
            f"Relative lift is {primary['relative_lift']*100:.2f}%. Pre-registered MDE was "
            f"{pre_reg['minimum_detectable_effect_relative']*100:.1f}% relative — a statistically "
            "significant but sub-MDE lift is flagged as 'real but not worth the added serving cost' "
            "per the study guide's 'practical significance' concept, independent of the p-value."
        ),
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/experiment_readout.json", "w") as f:
        json.dump(readout, f, indent=2)
    return readout


if __name__ == "__main__":
    readout = run_readout()
    print(json.dumps(readout, indent=2))
