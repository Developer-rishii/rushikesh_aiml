"""
fairness_audit.py — Stage C: "Quality, fairness and latency results for that tenant"

Computes, per protected_group (A vs B), on the held-out test set:
  - Demographic parity: P(top-10 shortlist | group) roughly equal across groups
  - Equal opportunity: P(top-10 | truly-would-hire, group) roughly equal
  - Selection rate ratio (4/5ths / 80% rule, a common regulatory heuristic)

protected_group is NEVER used as a model feature (see features.py) --
it is joined in here purely for the audit, exactly as the guide requires:
"fair across protected groups" is verified, not assumed.

Guide pitfall explicitly flagged: "A fairness audit done once, at the end,
as a formality." -- this script is designed to be re-run every training
cycle (see remediation.py) rather than run once and archived.
"""
import json
import os
import numpy as np
import pandas as pd
import joblib

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")
EXPER = os.path.join(os.path.dirname(__file__), "..", "experiments")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from features import FEATURE_COLUMNS


def main():
    test = pd.read_csv(f"{EXPER}/held_out_test_set.csv")
    model = joblib.load(f"{MODELS}/ranker_v1.joblib")
    test["model_score"] = model.predict(test[FEATURE_COLUMNS])

    K = 10
    top_k_flags = []
    for job_id, g in test.groupby("job_id"):
        g_sorted = g.sort_values("model_score", ascending=False)
        idx_top = g_sorted.index[:K]
        flags = pd.Series(0, index=g.index)
        flags.loc[idx_top] = 1
        top_k_flags.append(flags)
    test["in_top_k"] = pd.concat(top_k_flags)

    rows = []
    for grp, g in test.groupby("protected_group"):
        selection_rate = g["in_top_k"].mean()  # demographic parity numerator
        truly_qualified = g[g["hired"] == 1]
        equal_opportunity = (
            truly_qualified["in_top_k"].mean() if len(truly_qualified) else float("nan")
        )
        rows.append({
            "group": grp,
            "n": int(len(g)),
            "selection_rate_top10": round(float(selection_rate), 4),
            "equal_opportunity_rate": (
                round(float(equal_opportunity), 4) if not np.isnan(equal_opportunity) else None
            ),
        })

    df_rows = pd.DataFrame(rows).set_index("group")
    rates = df_rows["selection_rate_top10"]
    ratio = float(rates.min() / rates.max()) if rates.max() > 0 else None
    passes_4_5ths_rule = ratio is not None and ratio >= 0.8

    report = {
        "tenant": "AcmeFinServ_Pilot",
        "k": K,
        "per_group": rows,
        "demographic_parity_ratio_min_over_max": round(ratio, 4) if ratio is not None else None,
        "passes_4_5ths_rule": passes_4_5ths_rule,
        "note": (
            "4/5ths rule is a regulatory heuristic (min/max selection rate ratio >= 0.8), "
            "not a legal guarantee of fairness under India's DPDP or any specific hiring "
            "regulation. This audit must be re-run every retrain, not treated as a one-time "
            "sign-off (guide pitfall: 'fairness audit done once, at the end, as a formality')."
        ),
    }
    with open(f"{EXPER}/fairness_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
