"""
Stage B — "A configurable matching policy layer (weights/rules) per tenant"

Design: the ML model produces a tenant-agnostic base relevance score.
Each tenant has a versioned PolicyConfig that re-weights the SAME set of
transparent sub-signals (skill_overlap, experience_fit, distance_fit) and
can add bounded hard rules (e.g. min_skill_overlap). This is a deliberate
choice over "retrain a model per tenant" — see README.

Every config change is versioned and appended to an audit log (Stage B
"policy layer... versioned and auditable" + Pitfall "No model versioning").
"""
import json
import time
import numpy as np
from dataclasses import dataclass, asdict, field
from pathlib import Path
from config import DATA_DIR
AUDIT_LOG = DATA_DIR / "policy_audit_log.jsonl"
# ---- bounded config space (Stage: "Bounded configurability") ------------
BOUNDS = {
    "w_skill": (0.0, 1.0),
    "w_experience": (0.0, 1.0),
    "w_distance": (0.0, 1.0),
    "min_skill_overlap": (0.0, 0.9),   # can't require >90% overlap (nonsensical)
    "max_distance_km": (5.0, 500.0),
}
DEFAULT_CONFIG = {
    "w_skill": 0.5, "w_experience": 0.3, "w_distance": 0.2,
    "min_skill_overlap": 0.0, "max_distance_km": 500.0,
}


@dataclass
class PolicyConfig:
    tenant_id: str
    w_skill: float
    w_experience: float
    w_distance: float
    min_skill_overlap: float
    max_distance_km: float
    version: int = 1
    created_at: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)


def _exp_fit(exp_gap):
    return 1 / (1 + np.exp(-0.3 * exp_gap))


def _dist_fit(distance_km):
    return np.exp(-distance_km / 40)


def apply_policy(df, config: PolicyConfig, base_score_col="score"):
    """Combines the base ML score with tenant weights + hard rules.
    Returns df with a 'policy_score' column and an 'eligible' boolean
    (False = filtered out by a hard rule, e.g. max_distance_km)."""
    df = df.copy()
    exp_fit = _exp_fit(df["years_exp"] - df["req_years_exp"])
    dist_fit = _dist_fit(df["distance_km"])

    weighted = (config.w_skill * df["skill_overlap"] +
                config.w_experience * exp_fit +
                config.w_distance * dist_fit)
    # base ML score still anchors the ranking; policy re-weights on top
    # of it rather than replacing it, so the model's learned signal is
    # never fully discarded by a tenant config.
    df["policy_score"] = 0.5 * df[base_score_col] + 0.5 * weighted

    df["eligible"] = (
        (df["skill_overlap"] >= config.min_skill_overlap) &
        (df["distance_km"] <= config.max_distance_km)
    )
    df.loc[~df["eligible"], "policy_score"] = -np.inf
    return df


class PolicyStore:
    """In-memory + append-only-audit-log store of the LIVE config per
    tenant. Every write is versioned; nothing is ever silently overwritten."""

    def __init__(self):
        self._live = {}  # tenant_id -> PolicyConfig

    def get(self, tenant_id) -> PolicyConfig:
        if tenant_id not in self._live:
            self._live[tenant_id] = PolicyConfig(tenant_id=tenant_id, **DEFAULT_CONFIG)
        return self._live[tenant_id]

    def propose(self, tenant_id, overrides: dict) -> PolicyConfig:
        """Build a candidate config WITHOUT committing it — used by preview.py."""
        current = self.get(tenant_id)
        merged = current.to_dict()
        merged.update(overrides)
        merged.pop("version", None)
        merged.pop("created_at", None)
        merged.pop("tenant_id", None)
        return PolicyConfig(tenant_id=tenant_id, version=current.version + 1, **merged)

    def commit(self, config: PolicyConfig, actor="admin"):
        """Makes a previewed config live and appends an audit record.
        This is the ONLY function that changes what's served."""
        self._live[config.tenant_id] = config
        record = {"ts": time.time(), "actor": actor, **config.to_dict()}
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
        return config
