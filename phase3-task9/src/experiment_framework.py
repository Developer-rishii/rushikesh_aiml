"""
Stage B - Model variant serving behind the experiment framework
Stage C - A permanent holdout group for measuring cumulative model value

Design decisions (see DECISIONS.md for the full rejected-alternatives list):
  - Assignment is a deterministic hash of (user_id, salt) -> bucket in
    [0, BUCKETS). Same user + same experiment salt ALWAYS lands in the same
    bucket, on any server, any day, with no shared state required
    (consistent assignment, per study-guide core concept #2).
  - The holdout is PERMANENT: its bucket range is carved out first and never
    reused by any future experiment salt, so cumulative model value can be
    measured over months, not just one experiment's duration.
"""
import hashlib
from dataclasses import dataclass

BUCKETS = 10000
HOLDOUT_SALT = "placemux-permanent-holdout"   # fixed forever
EXPERIMENT_SALT = "exp-task9-ranker-v2"       # per-experiment

# Traffic split (task explicitly asks for a 10% candidate ramp).
HOLDOUT_PCT = 0.05      # never served ANY model - pure baseline-of-baselines
TREATMENT_PCT = 0.10    # candidate model v2
# remainder (85%) = control, served baseline_v1


def _bucket(key: str, salt: str) -> int:
    h = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()
    return int(h, 16) % BUCKETS


@dataclass
class Assignment:
    user_id: str
    is_holdout: bool
    variant: str  # "control" | "treatment" | "holdout"


class ExperimentFramework:
    """Deterministic variant assignment + guardrail-aware serving hook."""

    def __init__(self, holdout_pct: float = HOLDOUT_PCT, treatment_pct: float = TREATMENT_PCT):
        self.holdout_pct = holdout_pct
        self.treatment_pct = treatment_pct
        self.halted = False          # flips True if a guardrail trips
        self.halt_reason = None

    def assign(self, user_id: str) -> Assignment:
        # 1) Permanent holdout carve-out, fixed salt, independent of any
        #    experiment so it is stable across experiment restarts.
        holdout_bucket = _bucket(user_id, HOLDOUT_SALT)
        if holdout_bucket < self.holdout_pct * BUCKETS:
            return Assignment(user_id, True, "holdout")

        # 2) Remaining traffic split by the experiment-specific salt.
        exp_bucket = _bucket(user_id, EXPERIMENT_SALT)
        if exp_bucket < self.treatment_pct * BUCKETS:
            return Assignment(user_id, False, "treatment")
        return Assignment(user_id, False, "control")

    def halt(self, reason: str):
        self.halted = True
        self.halt_reason = reason

    def route_model(self, assignment: Assignment) -> str:
        """What model actually serves this request right now.

        Guardrail halt or holdout both route to the safe baseline - a
        halted experiment must never keep hurting users, and a holdout
        user must never see ANY experimental ranking by definition.
        """
        if assignment.variant == "holdout":
            return "baseline_v1"  # holdout still needs a UX; served baseline
        if self.halted:
            return "baseline_v1"
        if assignment.variant == "treatment":
            return "candidate_v2"
        return "baseline_v1"
