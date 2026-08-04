"""
model_registry.py — minimal model card generator so any decision can be
traced back to the exact model version (guide pitfall: "No model
versioning, so you cannot say which model produced a decision six
months ago.").
"""
import json
import os
import datetime

EXPER = os.path.join(os.path.dirname(__file__), "..", "experiments")
MODELS = os.path.join(os.path.dirname(__file__), "..", "models")


def main():
    with open(f"{EXPER}/experiment_log.json") as f:
        exp = json.load(f)
    with open(f"{EXPER}/metrics_offline.json") as f:
        metrics = json.load(f)
    with open(f"{EXPER}/fairness_report.json") as f:
        fairness = json.load(f)

    card = f"""# Model Card — {exp['model_version']}

**Tenant:** AcmeFinServ_Pilot
**Created:** {datetime.date.today().isoformat()}
**Artifact:** models/{exp['model_version']}.joblib

## Training data
- Rows trained on: {exp['trained_on_rows']}
- Held-out rows: {exp['held_out_rows']} across jobs {exp['held_out_jobs']}
- Features: {', '.join(exp['features'])}
- Target: {exp['target']}
- Random seed: {exp['reproducible_seed']} (fully reproducible)

## Approach & rejected alternatives
{exp['chosen_approach']}

## Offline evaluation vs baseline ({exp['baseline']})
- nDCG@10 delta: {metrics['offline_gain']['nDCG@10_delta']}
- MAP@10 delta: {metrics['offline_gain']['MAP@10_delta']}
- Precision@10 delta: {metrics['offline_gain']['Precision@10_delta']}
- Online proxy (hire-capture@10) delta: {metrics['online_proxy_hire_capture_at_10']['delta']}

## Fairness (protected_group A vs B)
- Demographic parity ratio (min/max): {fairness['demographic_parity_ratio_min_over_max']}
- Passes 4/5ths rule: {fairness['passes_4_5ths_rule']}
- See experiments/fairness_report.json for equal-opportunity gap finding.

## Known limitations (see docs/remediation_list.md for full list)
- Pointwise proxy, not listwise LambdaMART (LightGBM unavailable in build env)
- protected_group excluded from features by design, used only for audit
- Trained on synthetic-but-realistic tenant data, not a live enterprise export

## Serving fallback
If this artifact is missing or fails to load, the serving layer falls back
to skill_overlap baseline ranking (verified in experiments/latency_report.json
chaos test) rather than failing the request.
"""
    with open(f"{MODELS}/model_card.md", "w") as f:
        f.write(card)
    print("Wrote model_card.md")


if __name__ == "__main__":
    main()
