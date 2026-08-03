"""
Stage C — "Guardrails preventing configurations that are unfair or
nonsensical"

Two independent checks, both must pass before policy.PolicyStore.commit()
is allowed to run (enforced in serve.py / demo.py, never bypassable from
the admin UI layer):

1. STRUCTURAL guardrail (nonsensical configs): weights don't sum sanely,
   values out of bounds, hard rule that filters out ~everyone.
2. FAIRNESS guardrail (unfair configs): re-scores a held-out labelled
   sample under the proposed config and checks the selection-rate parity
   between protected groups (using the 4/5ths / disparate-impact rule),
   WITHOUT ever using the protected attribute as a feature — it is used
   only to *audit* the outcome, exactly as DPDP/fair-hiring practice
   requires.
"""
from dataclasses import dataclass
from typing import List
import numpy as np

from policy import BOUNDS, apply_policy, PolicyConfig

FOUR_FIFTHS_THRESHOLD = 0.8  # standard disparate-impact cutoff


@dataclass
class GuardrailResult:
    passed: bool
    violations: List[str]


def check_structural(config: PolicyConfig) -> GuardrailResult:
    violations = []
    for field_name, (lo, hi) in BOUNDS.items():
        val = getattr(config, field_name)
        if not (lo <= val <= hi):
            violations.append(f"{field_name}={val} outside allowed bounds [{lo}, {hi}]")

    total_w = config.w_skill + config.w_experience + config.w_distance
    if not (0.5 <= total_w <= 1.5):
        violations.append(f"weight sum {total_w:.2f} is nonsensical (expected ~1.0)")

    if config.min_skill_overlap > 0.85:
        violations.append("min_skill_overlap so high it would reject nearly all candidates")

    return GuardrailResult(passed=len(violations) == 0, violations=violations)


def check_fairness(config: PolicyConfig, audit_df, base_score_col="score",
                    top_k=10, protected_col="gender_proxy"):
    """Simulate the config on a held-out audit sample (per job, top-k
    'shortlisted by policy') and check the four-fifths rule between
    protected groups. protected_col is used ONLY here, never as a model
    or policy feature."""
    scored = apply_policy(audit_df, config, base_score_col)
    violations = []
    selection_rates = {}

    picks = []
    for job_id, g in scored.groupby("job_id"):
        top = g.sort_values("policy_score", ascending=False).head(top_k)
        picks.append(top)
    picks = np.concatenate([p.index.values for p in picks]) if picks else np.array([])
    selected_mask = scored.index.isin(picks)

    tprs = {}
    
    for grp, g in scored.groupby(protected_col):
        rate = selected_mask[g.index].mean()
        selection_rates[grp] = float(rate)
        
        # Equal-opportunity parity (TPR parity):
        # Formula: TPR_g = P(selected=1 | label>=0.5, group=g)
        # We consider a candidate 'qualified' (Y=1) if their true label >= 0.5.
        # We check if the True Positive Rate is similar across groups.
        qualified_mask = g["label"] >= 0.5
        if qualified_mask.sum() > 0:
            tpr = selected_mask[g.index][qualified_mask].mean()
            tprs[grp] = float(tpr)

    if len(selection_rates) >= 2:
        rates = list(selection_rates.values())
        ratio = min(rates) / max(rates) if max(rates) > 0 else 1.0
        if ratio < FOUR_FIFTHS_THRESHOLD:
            violations.append(
                f"disparate impact: selection-rate ratio {ratio:.2f} < "
                f"{FOUR_FIFTHS_THRESHOLD} four-fifths threshold "
                f"(rates={selection_rates})")
                
    if len(tprs) >= 2:
        tpr_vals = list(tprs.values())
        tpr_ratio = min(tpr_vals) / max(tpr_vals) if max(tpr_vals) > 0 else 1.0
        if tpr_ratio < FOUR_FIFTHS_THRESHOLD:
            violations.append(
                f"equal opportunity violation: TPR ratio {tpr_ratio:.2f} < "
                f"{FOUR_FIFTHS_THRESHOLD} threshold "
                f"(TPRs={tprs})")

    return GuardrailResult(passed=len(violations) == 0, violations=violations), selection_rates


def validate_config(config: PolicyConfig, audit_df, base_score_col="score"):
    """Single entry point: both checks must pass. This is what
    PolicyStore.commit() must be gated behind."""
    structural = check_structural(config)
    fairness, rates = check_fairness(config, audit_df, base_score_col)
    all_violations = structural.violations + fairness.violations
    return GuardrailResult(passed=(structural.passed and fairness.passed),
                            violations=all_violations), rates
