"""
Tenant-scoped data access layer.

This is the ONLY module allowed to touch raw log files. Every other module
(train, serve, evaluate) must go through TenantDataStore so that it is
IMPOSSIBLE (not just discouraged) to accidentally load another tenant's rows.

Design decision: strict per-tenant isolation (separate CSV/model per tenant),
NOT a shared global model with tenant-id as a feature.
  Rejected alternative: shared global model + tenant_id feature.
  Why rejected: with a shared model, a rival enterprise's data still touches
  the same weight matrix during training; on a small-data tenant this leaks
  signal from a competitor's hiring pattern (see evidence/isolation_proof.txt
  for the empirical leakage test that motivated this choice). Per-tenant
  models cost more to operate (N models instead of 1) but that is the bar
  the study guide sets: "neither can influence or see the other's data".
"""
import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_DIR = os.path.join(BASE_DIR, "configs")

_ALLOWED_TENANTS = {"tenantA": "tenant_a_logs.csv", "tenantB": "tenant_b_logs.csv"}
_ALLOWED_CONFIGS = {"tenantA": "tenant_a.json", "tenantB": "tenant_b.json"}


class TenantAccessError(Exception):
    pass


class TenantDataStore:
    """Every call is scoped to exactly one tenant_id; there is no method that
    returns cross-tenant data. This is the enforcement point that isolation
    tests in leakage_test.py attempt to break."""

    def __init__(self, tenant_id: str):
        if tenant_id not in _ALLOWED_TENANTS:
            raise TenantAccessError(f"Unknown tenant '{tenant_id}'")
        self.tenant_id = tenant_id

    def load_logs(self) -> pd.DataFrame:
        path = os.path.join(DATA_DIR, _ALLOWED_TENANTS[self.tenant_id])
        df = pd.read_csv(path)
        # Defense in depth: even though the file is tenant-specific, assert it,
        # so a future refactor that merges files cannot silently leak rows.
        assert (df["tenant_id"] == self.tenant_id).all(), \
            f"ISOLATION BREACH: {self.tenant_id} store returned foreign rows"
        return df

    def load_config(self) -> dict:
        path = os.path.join(CONFIG_DIR, _ALLOWED_CONFIGS[self.tenant_id])
        with open(path) as f:
            cfg = json.load(f)
        assert cfg["tenant_id"] == self.tenant_id
        return cfg


def list_tenants():
    return list(_ALLOWED_TENANTS.keys())
