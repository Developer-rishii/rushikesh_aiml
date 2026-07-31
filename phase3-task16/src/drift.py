"""
Minimal drift monitor: compares today's score distribution to the training-time
score distribution using population stability index (PSI). Meant to run on a
schedule in production; here it's demoed as a callable check per tenant.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isolation import TenantDataStore, list_tenants
from serve import TenantInferenceService
from features import compute_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def psi(expected, actual, bins=10):
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.quantile(expected, quantiles)
    cuts[0], cuts[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)
    e_pct = np.clip(e_counts / max(len(expected), 1), 1e-4, None)
    a_pct = np.clip(a_counts / max(len(actual), 1), 1e-4, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def check_tenant(tenant_id, simulate_shift=False):
    store = TenantDataStore(tenant_id)
    df = store.load_logs()
    svc = TenantInferenceService(tenant_id)
    X = compute_features(df)
    train_time_scores = svc._model_bundle["model"].predict_proba(X)[:, 1]

    if simulate_shift:
        # simulate a serving-time population shift (e.g. new job category mix)
        current_scores = train_time_scores * 0.5 + np.random.RandomState(1).uniform(0, 0.3, len(train_time_scores))
    else:
        current_scores = train_time_scores

    score = psi(train_time_scores, current_scores)
    flag = "ALERT" if score > 0.25 else ("WATCH" if score > 0.1 else "OK")
    return {"tenant_id": tenant_id, "psi": round(score, 4), "flag": flag}


if __name__ == "__main__":
    results = []
    for t in list_tenants():
        results.append(check_tenant(t, simulate_shift=False))
    results.append(check_tenant("tenantA", simulate_shift=True))
    for r in results:
        print(json.dumps(r))
    with open(os.path.join(BASE_DIR, "evidence", "drift_report.json"), "w") as f:
        json.dump(results, f, indent=2)
