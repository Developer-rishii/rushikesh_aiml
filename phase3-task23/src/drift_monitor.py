"""
Detects train/serve skew (Section 5's 'single biggest silent killer').
Compares the distribution of a feature as computed at training time vs how
it would be computed at serving time, using a PSI (Population Stability Index).
PSI > 0.25 = severe skew -> should block deploy.
"""
import pandas as pd, numpy as np, json, os
from datetime import datetime, timezone

BASE = os.path.join(os.path.dirname(__file__), "..")
df = pd.read_csv(f"{BASE}/data/interactions.csv")

def psi(expected, actual, bins=10):
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.quantile(expected, quantiles)
    cuts[0], cuts[-1] = -np.inf, np.inf
    e_counts, _ = np.histogram(expected, bins=cuts)
    a_counts, _ = np.histogram(actual, bins=cuts)
    e_pct = np.clip(e_counts / len(expected), 1e-6, None)
    a_pct = np.clip(a_counts / len(actual), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

# Simulate the recency feature: training pipeline computes days, serving path
# (per the injected bug in generate_data.py) computes hours under the same field name.
train_vals = df["recency_feature_train"].values
serve_vals_buggy = df["recency_feature_serve"].values  # hours, NOT days -> huge skew
serve_vals_fixed = train_vals + np.random.default_rng(1).normal(0, 1, len(train_vals))  # what it SHOULD look like

result = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "feature": "recency_feature",
    "psi_buggy_serving_path": round(psi(train_vals, serve_vals_buggy), 4),
    "psi_healthy_serving_path": round(psi(train_vals, serve_vals_fixed), 4),
    "verdict_buggy_path": "BLOCK DEPLOY - severe train/serve skew detected" if psi(train_vals, serve_vals_buggy) > 0.25 else "OK",
    "verdict_healthy_path": "OK" if psi(train_vals, serve_vals_fixed) < 0.1 else "REVIEW",
    "note": "This demonstrates the monitor correctly catches the deliberately-injected "
            "hours-vs-days unit bug (Section 5), and does NOT false-positive on a healthy feature.",
}
os.makedirs(f"{BASE}/audit", exist_ok=True)
json.dump(result, open(f"{BASE}/audit/drift_check.json", "w"), indent=2)
print(json.dumps(result, indent=2))
