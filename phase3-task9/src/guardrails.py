"""
Stage D - Guardrail metrics that halt a bad model.

A guardrail is checked once per day, per variant, against the CONTROL
(baseline) as the reference point - not against an arbitrary constant -
because "good" drifts with seasonality and a fixed constant would either
never fire or fire on noise.

Halt rule: candidate must not fall more than `tolerance` below control on
ANY guardrail metric for `consecutive_days_to_halt` days running. This
answers the brainstorming question directly: "which guardrail must never be
crossed, even for a big win?" -> conversion_rate and fairness_gap are hard
guardrails; CTR alone is not (a model can legitimately trade a little CTR
for more relevant, lower-volume matches).
"""
from dataclasses import dataclass, field
import pandas as pd

HARD_GUARDRAILS = ["conversion_rate", "fairness_gap"]
SOFT_GUARDRAILS = ["CTR"]
TOLERANCE = {
    "conversion_rate": 0.10,   # candidate may not be >10% relatively worse
    "fairness_gap": 0.05,      # absolute selection-rate gap must stay <=5pp
    "CTR": 0.15,
}
CONSECUTIVE_DAYS_TO_HALT = 2


@dataclass
class GuardrailDay:
    day: int
    breaches: list = field(default_factory=list)

    @property
    def breached(self) -> bool:
        return len(self.breaches) > 0


def evaluate_day(day: int, control_metrics: dict, treatment_metrics: dict,
                  fairness_gap_treatment: float) -> GuardrailDay:
    breaches = []

    # relative comparison for rate metrics
    for m in ["conversion_rate", "CTR"]:
        c, t = control_metrics.get(m, 0), treatment_metrics.get(m, 0)
        if c <= 0:
            continue
        rel_drop = (c - t) / c
        if rel_drop > TOLERANCE[m]:
            breaches.append(
                f"{m}: treatment {t:.4f} vs control {c:.4f} "
                f"({rel_drop*100:.1f}% relative drop, tolerance {TOLERANCE[m]*100:.0f}%)"
            )

    # absolute comparison for fairness (a gap, not a rate)
    if fairness_gap_treatment > TOLERANCE["fairness_gap"]:
        breaches.append(
            f"fairness_gap: {fairness_gap_treatment:.4f} exceeds hard limit "
            f"{TOLERANCE['fairness_gap']:.4f}"
        )

    return GuardrailDay(day=day, breaches=breaches)


def should_halt(daily_results: list) -> tuple:
    """Halt if the LAST N consecutive days all breached a HARD guardrail."""
    if len(daily_results) < CONSECUTIVE_DAYS_TO_HALT:
        return False, None
    last_n = daily_results[-CONSECUTIVE_DAYS_TO_HALT:]
    hard_breach_days = [
        d for d in last_n
        if any(any(hg in b for hg in HARD_GUARDRAILS) for b in d.breaches)
    ]
    if len(hard_breach_days) == CONSECUTIVE_DAYS_TO_HALT:
        reasons = "; ".join(b for d in hard_breach_days for b in d.breaches)
        return True, reasons
    return False, None
