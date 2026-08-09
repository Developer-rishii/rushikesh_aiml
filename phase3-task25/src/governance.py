"""
Certification pack - governance section (Sec 12: "No model versioning, so
you cannot say which model produced a decision six months ago" is a named
failure). Generates a model card from the actual registry entry, not from
memory, so it can never drift out of sync with what was really trained.
"""
import os, json, time

ROOT = os.path.dirname(os.path.dirname(__file__))

def main():
    registry = json.load(open(f"{ROOT}/registry/model_registry.json"))
    entry = registry[-1]
    fairness = json.load(open(f"{ROOT}/reports/fairness_audit.json"))
    offline = json.load(open(f"{ROOT}/reports/offline_eval.json"))
    latency = json.load(open(f"{ROOT}/reports/latency_cost.json"))

    card = f"""# Model Card - PlaceMux Ranking Model {entry['version']}

**Trained:** {time.ctime(entry['trained_at_unix'])}
**Owner:** AI/ML Engineering, Sprint E
**Training data:** {entry['train_rows']} logged rows (day<20), held out {entry['test_rows']} rows (day>=20)
**Data split:** {entry['data_split']}
**Features:** {', '.join(entry['features'])}
**Objective:** LambdaMART (lambdarank), NDCG@10

## Intended use
Ranks job postings for a candidate at impression time. NOT used to make
autonomous hire/reject decisions - always shown to a human recruiter/candidate.

## Known limitations
- `exp_years` field is vulnerable to a serving-side under-logging bug
  (caught by drift_rollback.py's PSI monitor, see rollback_decision.json).
- Precision@10 lift over baseline is ~0 (see offline_eval.json) - the win is
  concentrated in ranking order (nDCG/MAP), not in raw relevant-item recall.
  Flagged, not hidden.

## Fairness
Demographic parity gap: {fairness['demographic_parity_gap']} (threshold {fairness['threshold']}, {'PASS' if fairness['pass_demographic_parity'] else 'FAIL'})
Equal opportunity gap: {fairness['equal_opportunity_gap']} (threshold {fairness['threshold']}, {'PASS' if fairness['pass_equal_opportunity'] else 'FAIL'})

## Serving
p95 latency: {latency['p95_ms']} ms (SLO {latency['slo_target_p95_ms']} ms - {'MET' if latency['slo_met'] else 'MISSED'})
Estimated cost: ${latency['estimated_cost_usd_per_1000_requests']} / 1000 requests

## Versioning & rollback
Model artifact: `{entry['model_path']}`
Registry: `registry/model_registry.json` (append-only, every training run logged)
Rollback trigger: PSI(exp_years) > 0.25 vs training distribution -> revert to previous version within 5 min.
"""
    open(f"{ROOT}/reports/model_card.md", "w").write(card)
    print(card)

if __name__ == "__main__":
    main()
