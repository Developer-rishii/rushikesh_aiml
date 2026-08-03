"""
Stage D — "An admin-facing preview of a config's effect before it goes live"

Given a proposed config, shows:
  - guardrail verdict (pass/fail + reasons)
  - before/after top-10 ranking for a sample job (does the actual order change?)
  - funnel-level impact estimate: how many previously-eligible candidates
    would now be filtered out, and the fairness selection-rate delta
This is what stands between "admin fat-fingers a config" and "funnel breaks
in prod" (Pitfall: "No preview or rollback of a config change").
"""
import pandas as pd
from policy import apply_policy
from guardrails import validate_config, check_fairness


def preview_config(store, tenant_id, overrides, audit_df, sample_job_id=None,
                    base_score_col="score"):
    proposed = store.propose(tenant_id, overrides)
    current = store.get(tenant_id)

    result, rates = validate_config(proposed, audit_df, base_score_col)

    tenant_rows = audit_df[audit_df.tenant_id == tenant_id]
    if sample_job_id is None and len(tenant_rows):
        sample_job_id = tenant_rows.job_id.iloc[0]

    before_top, after_top = None, None
    if sample_job_id is not None:
        job_rows = audit_df[audit_df.job_id == sample_job_id]
        before = apply_policy(job_rows, current, base_score_col)
        after = apply_policy(job_rows, proposed, base_score_col)
        before_top = before.sort_values("policy_score", ascending=False)[
            ["candidate_id", "policy_score", "eligible"]].head(10)
        after_top = after.sort_values("policy_score", ascending=False)[
            ["candidate_id", "policy_score", "eligible"]].head(10)

    # funnel impact: eligibility change across ALL rows for this tenant
    before_all = apply_policy(tenant_rows, current, base_score_col)
    after_all = apply_policy(tenant_rows, proposed, base_score_col)
    funnel_impact = {
        "eligible_before": int(before_all.eligible.sum()),
        "eligible_after": int(after_all.eligible.sum()),
        "pct_change": round(
            100 * (after_all.eligible.sum() - before_all.eligible.sum())
            / max(1, before_all.eligible.sum()), 1),
    }

    return {
        "tenant_id": tenant_id,
        "proposed_config": proposed.to_dict(),
        "guardrail_passed": result.passed,
        "guardrail_violations": result.violations,
        "fairness_selection_rates": rates,
        "funnel_impact": funnel_impact,
        "sample_job_id": sample_job_id,
        "before_top10": before_top,
        "after_top10": after_top,
    }, proposed
