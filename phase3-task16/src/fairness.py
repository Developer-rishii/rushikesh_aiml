"""
Fairness check per tenant. Pitfall called out explicitly in the study guide:
'A fairness audit done once, at the end, as a formality.' -> this is written
as a reusable script meant to be re-run every retrain, not a one-off notebook.

Metric: demographic parity difference in shortlist rate between protected_group
A and B, at each tenant's OWN configured threshold.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isolation import TenantDataStore, list_tenants
from serve import TenantInferenceService

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def audit_tenant(tenant_id):
    store = TenantDataStore(tenant_id)
    df = store.load_logs()
    svc = TenantInferenceService(tenant_id)
    ranked, _ = svc.rank_candidates(df.assign())  # score all rows, no cap effect on rate calc
    # recompute without the service's head(cap) truncation for a fair population-level rate
    from features import compute_features
    X = compute_features(df)
    proba = svc._model_bundle["model"].predict_proba(X)[:, 1]
    df = df.copy()
    df["score"] = proba
    df["shortlisted"] = df["score"] >= svc.cfg["shortlist_threshold"]

    rates = df.groupby("protected_group")["shortlisted"].mean()
    parity_gap = float(abs(rates.get("A", 0) - rates.get("B", 0)))
    return {
        "tenant_id": tenant_id,
        "shortlist_rate_group_A": round(float(rates.get("A", 0)), 4),
        "shortlist_rate_group_B": round(float(rates.get("B", 0)), 4),
        "demographic_parity_gap": round(parity_gap, 4),
        "flag": "REVIEW" if parity_gap > 0.10 else "OK",
    }


if __name__ == "__main__":
    import json
    results = [audit_tenant(t) for t in list_tenants()]
    for r in results:
        print(json.dumps(r, indent=2))
    with open(os.path.join(BASE_DIR, "evidence", "fairness_report.json"), "w") as f:
        json.dump(results, f, indent=2)
