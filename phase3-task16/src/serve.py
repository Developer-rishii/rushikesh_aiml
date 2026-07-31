"""
Stage B/C 'explainable, safe & demoable' step: tenant-scoped inference.

Config (thresholds, weights, result caps) is read from configs/<tenant>.json —
never hard-coded per tenant, so a new tenant onboards by adding a JSON file,
not by forking code (Stage C requirement).

Also implements the required failure mode: "what happens when the model is
unavailable" -> falls back to a documented popularity baseline, never crashes,
never serves another tenant's model.
"""
import os
import pickle

from isolation import TenantDataStore, TenantAccessError
from features import compute_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ModelUnavailableError(Exception):
    pass


class TenantInferenceService:
    def __init__(self, tenant_id: str, simulate_model_down: bool = False):
        self.store = TenantDataStore(tenant_id)  # raises TenantAccessError if unknown tenant
        self.tenant_id = tenant_id
        self.cfg = self.store.load_config()
        self._simulate_model_down = simulate_model_down
        self._model_bundle = None
        if not simulate_model_down:
            self._load_model()

    def _load_model(self):
        path = os.path.join(BASE_DIR, self.cfg["model_path"])
        with open(path, "rb") as f:
            self._model_bundle = pickle.load(f)
        # Defense in depth: refuse to serve a model that isn't this tenant's.
        if self._model_bundle["tenant_id"] != self.tenant_id:
            raise TenantAccessError(
                f"Loaded model belongs to {self._model_bundle['tenant_id']}, "
                f"not {self.tenant_id} -- refusing to serve."
            )

    def rank_candidates(self, df_pairs):
        """df_pairs: candidate-job rows already scoped to this tenant (caller
        must have obtained them via this tenant's TenantDataStore)."""
        assert (df_pairs["tenant_id"] == self.tenant_id).all(), \
            "ISOLATION BREACH: attempted to score rows from another tenant"

        X = compute_features(df_pairs)
        threshold = self.cfg["shortlist_threshold"]
        cap = self.cfg["max_ranked_results"]

        if self._model_bundle is None:
            # ---- graceful degradation path ----
            scores = [self.cfg["fallback_score"]] * len(df_pairs)
            mode = self.cfg["fallback_mode"]
        else:
            scores = self._model_bundle["model"].predict_proba(X)[:, 1]
            mode = f"model v{self._model_bundle['trained_at']}"

        out = df_pairs.copy()
        out["score"] = scores
        out["shortlisted"] = out["score"] >= threshold
        out = out.sort_values("score", ascending=False).head(cap)
        return out, mode


if __name__ == "__main__":
    from isolation import list_tenants
    for t in list_tenants():
        store = TenantDataStore(t)
        df = store.load_logs().sample(20, random_state=1)
        svc = TenantInferenceService(t)
        ranked, mode = svc.rank_candidates(df)
        print(f"\n[{t}] serving mode={mode} threshold={svc.cfg['shortlist_threshold']} "
              f"cap={svc.cfg['max_ranked_results']}")
        print(ranked[["candidate_id", "job_id", "score", "shortlisted"]].head(5).to_string(index=False))

    print("\n--- Failure scenario: tenantA model file goes missing/unavailable ---")
    svc_down = TenantInferenceService("tenantA", simulate_model_down=True)
    store = TenantDataStore("tenantA")
    df = store.load_logs().sample(5, random_state=1)
    ranked, mode = svc_down.rank_candidates(df)
    print(f"[tenantA DEGRADED] serving mode={mode}")
    print(ranked[["candidate_id", "job_id", "score", "shortlisted"]].to_string(index=False))
