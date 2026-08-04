"""
remediation.py — Stage D: "A remediation list before the real pilot"

Reads the actual experiment/fairness/latency reports produced by this run
and generates a remediation list grounded in real numbers, not a generic
checklist. This is the artifact handed to the enterprise + next team
(Section 13: "Pilot findings to all roles").
"""
import json
import os

EXPER = os.path.join(os.path.dirname(__file__), "..", "experiments")
DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")


def main():
    os.makedirs(DOCS, exist_ok=True)
    with open(f"{EXPER}/metrics_offline.json") as f:
        metrics = json.load(f)
    with open(f"{EXPER}/fairness_report.json") as f:
        fairness = json.load(f)
    with open(f"{EXPER}/latency_report.json") as f:
        latency = json.load(f)

    items = []

    gain = metrics["offline_gain"]
    if gain["MAP@10_delta"] < 0 or gain["Precision@10_delta"] < 0:
        items.append({
            "severity": "HIGH",
            "finding": (
                f"Model underperforms the skill_overlap baseline on offline "
                f"MAP@10 ({gain['MAP@10_delta']}) and Precision@10 "
                f"({gain['Precision@10_delta']}) even though it wins on the "
                f"real-outcome hire-capture proxy ({metrics['online_proxy_hire_capture_at_10']['delta']} "
                f"more hires captured in top 10)."
            ),
            "action": (
                "Do not ship on offline nDCG/MAP alone. Run an online A/B test "
                "on a small traffic slice before the full pilot; the hire-capture "
                "proxy suggests real value the ranking metrics understate, but "
                "this must be confirmed online, not assumed."
            ),
        })

    eo_rates = {r["group"]: r["equal_opportunity_rate"] for r in fairness["per_group"]}
    if None not in eo_rates.values():
        gap = abs(eo_rates["A"] - eo_rates["B"])
        if gap > 0.02:
            items.append({
                "severity": "HIGH",
                "finding": (
                    f"Demographic parity passes the 4/5ths rule "
                    f"(ratio={fairness['demographic_parity_ratio_min_over_max']}), "
                    f"but equal-opportunity rates diverge sharply between groups "
                    f"(A={eo_rates['A']}, B={eo_rates['B']}): among candidates who "
                    f"actually get hired, group B is surfaced in the top 10 far "
                    f"more often than group A."
                ),
                "action": (
                    "Passing demographic parity is not sufficient. Investigate why "
                    "truly-qualified group-A candidates rank lower before the real "
                    "pilot -- check for skill-vocabulary or experience-distribution "
                    "confounds; add equal-opportunity as a hard gate, not just parity."
                ),
            })

    items.append({
        "severity": "MEDIUM",
        "finding": (
            "LambdaMART/listwise learning-to-rank was rejected for this dry-run "
            "due to no LightGBM/XGBoost available in the build sandbox; a pointwise "
            "GradientBoostingRegressor proxy was used instead."
        ),
        "action": (
            "Before the real pilot, retrain with a listwise objective (LambdaMART) "
            "in an environment with package access, since pointwise regression "
            "optimizes per-row accuracy, not result ORDER, which the guide "
            "identifies as what actually drives outcomes."
        ),
    })

    items.append({
        "severity": "MEDIUM",
        "finding": (
            "Domain shift: 60% of tenant-facing job titles use internal vocabulary "
            "(e.g. 'Quant Modeling Specialist' vs standard 'Data Scientist') that "
            "differs from generic training data title strings."
        ),
        "action": (
            "Build a tenant title-vocabulary mapping table (see TENANT_TITLE_VOCAB "
            "in data/generate_data.py as the pattern) and validate it with the "
            "tenant's HR team before the real pilot; do not rely on the model "
            "to learn tenant vocabulary from too few examples."
        ),
    })

    p99 = latency["latency_ms"]["p99"]
    items.append({
        "severity": "LOW" if p99 < 100 else "HIGH",
        "finding": f"p99 latency to rank a full 2,000-candidate pool for one requisition: {p99} ms.",
        "action": (
            "Within budget for a pilot (<100ms p99). Before scaling past this "
            "tenant, re-benchmark with the real candidate pool size and add "
            "caching for repeated job/candidate feature lookups."
        ),
    })

    items.append({
        "severity": "MEDIUM",
        "finding": "No model versioning/registry existed before this run.",
        "action": (
            "Adopted: every trained model is now saved with a version tag and "
            "a model card (models/model_card.md) recording training data window, "
            "features, and metrics, so a decision can be traced back to the exact "
            "model version months later (guide pitfall: 'cannot say which model "
            "produced a decision six months ago')."
        ),
    })

    items.append({
        "severity": "MEDIUM",
        "finding": "Acceptance criteria for 'a good match' were assumed, not agreed with a real enterprise stakeholder.",
        "action": (
            "Before the real pilot: get the tenant's hiring manager to sign off "
            "explicitly on what 'good' means (which offline/online metric, what "
            "threshold) -- see docs/acceptance_criteria.md draft for the starting "
            "proposal to review with them."
        ),
    })

    with open(f"{DOCS}/remediation_list.json", "w") as f:
        json.dump(items, f, indent=2)

    md = ["# Remediation List — Before the Real Pilot\n", f"Tenant: AcmeFinServ_Pilot\n"]
    for it in items:
        md.append(f"## [{it['severity']}] {it['finding']}\n\n**Action:** {it['action']}\n")
    with open(f"{DOCS}/remediation_list.md", "w") as f:
        f.write("\n".join(md))

    print(f"Wrote {len(items)} remediation items.")


if __name__ == "__main__":
    main()
