"""
fairness_audit.py
-------------------
Pitfall explicitly called out in the study guide: "A fairness audit done
once, at the end, as a formality." -- and prerequisite: "Familiarity with
the fairness and DPDP constraints on automated hiring decisions."

FIX: we run this audit at TWO separate points against a synthetic protected-
like attribute (`fairness_group`, carried through the pipeline but NEVER fed
into the model as a feature -- see feature_engineering.py) and store both
results with timestamps, so this can be wired into CI to run on every
retrain rather than once manually:
  1. On the BASELINE rule (does the simple heuristic already have a gap?)
  2. On the FINAL model's holdout predictions (did the model make it worse
     or better?)

Metrics:
  - Demographic parity difference: |P(flagged=1 | group_A) - P(flagged=1 | group_B)|
  - Equal opportunity difference: |TPR(group_A) - TPR(group_B)| computed only
    among candidates who actually churned (the label-conditional version --
    this is the one that matters most for a hiring-adjacent decision, since
    it asks "among people who truly needed the intervention, did both groups
    get flagged at the same rate?").

DPDP note: `fairness_group` here is a SIMULATED stand-in attribute (not a
real protected characteristic), used purely to exercise this audit
mechanism end-to-end. In a real deployment this would be run against actual
legally-protected classes with legal/DPDP sign-off on what may even be
collected for audit purposes, and audit results would never be joined back
into the serving path.
"""
import json
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from feature_engineering import FEATURE_COLUMNS
from baseline_model import rule_14_day_binary

ROOT = Path(__file__).resolve().parents[1]


def _fairness_metrics(df, flag_col, label_col="churned"):
    out = {}
    rates = df.groupby("fairness_group")[flag_col].mean().to_dict()
    out["flag_rate_by_group"] = rates
    out["demographic_parity_diff"] = float(abs(rates.get("group_A", 0) - rates.get("group_B", 0)))

    churned = df[df[label_col] == 1]
    if len(churned):
        tpr = churned.groupby("fairness_group")[flag_col].mean().to_dict()
    else:
        tpr = {}
    out["tpr_by_group_among_actual_churners"] = tpr
    if "group_A" in tpr and "group_B" in tpr:
        out["equal_opportunity_diff"] = float(abs(tpr["group_A"] - tpr["group_B"]))
    else:
        out["equal_opportunity_diff"] = None
    return out


def run_audit():
    holdout = pd.read_csv(ROOT / "data/processed/holdout_snapshots.csv")

    # Stage 1 audit: the baseline rule
    holdout["baseline_flag"] = rule_14_day_binary(holdout, threshold=14)
    baseline_audit = _fairness_metrics(holdout, "baseline_flag")

    # Stage 2 audit: the trained model, at the same operating threshold used in evaluate.py
    with open(ROOT / "models/churn_model_v1.pkl", "rb") as f:
        model = pickle.load(f)
    scores = model.predict_proba(holdout[FEATURE_COLUMNS])[:, 1]
    thresh = scores.mean() + scores.std()  # simple consistent cutoff for audit purposes
    import numpy as np
    thresh = np.percentile(scores, 90)
    holdout["model_flag"] = (scores >= thresh).astype(int)
    model_audit = _fairness_metrics(holdout, "model_flag")

    report = {
        "audit_timestamp": datetime.now().isoformat(),
        "note": "fairness_group is a SIMULATED stand-in attribute, not fed to the model as a feature",
        "stage_1_baseline_rule_audit": baseline_audit,
        "stage_2_trained_model_audit": model_audit,
        "recommendation": (
            "Re-run this exact audit on every retrain (wire into the CI/experiment_log step), "
            "not just once at project end. Flag for manual review if demographic_parity_diff "
            "or equal_opportunity_diff exceeds 0.10."
        ),
    }
    (ROOT / "outputs").mkdir(parents=True, exist_ok=True)
    out_path = ROOT / "outputs" / "fairness_audit_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_audit()
