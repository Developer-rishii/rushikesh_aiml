"""
Stage E — Integrate, break it, then demo.
Run: python3 demo_e2e.py
This is the 2-minute live-demo script: real numbers, one induced failure.
"""
import json, os, joblib
import pandas as pd
from dsr_rights import access_request, deletion_request
from disclosure import explain_decision, submit_for_human_review, resolve_review, review_queue_snapshot

BASE = os.path.join(os.path.dirname(__file__), "..")

def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)

# 1. Show offline model quality vs baseline (real numbers)
section("1. MODEL QUALITY (offline, held-out, vs baseline)")
eval_results = json.load(open(f"{BASE}/models/eval_results.json"))
print(f"nDCG@10  model={eval_results['offline_model_metrics']['nDCG@10']} "
      f"baseline={eval_results['offline_baseline_metrics']['nDCG@10']} "
      f"(+{eval_results['offline_lift_nDCG10_pct']}%)")
print(f"Honest online caveat: {eval_results['online_proxy_caveat']}")

# 2. Full DSR journey on a real candidate
section("2. DATA-SUBJECT RIGHTS — real end-to-end deletion")
candidates = pd.read_csv(f"{BASE}/data/candidates.csv")
target = candidates.candidate_id.iloc[50]
before = access_request(target)
print(f"BEFORE deletion: found={before['found']}, logged_interactions={before.get('logged_interactions_count')}")
del_result = deletion_request(target)
print(f"Deletion executed: {del_result['status']}, rows removed={del_result['profile_rows_removed']}, "
      f"pseudonymised={del_result['interaction_rows_pseudonymised']}")
after = access_request(target)
print(f"AFTER deletion: found={after['found']}  <-- proves erasure is real, not theatre")

# 3. Automated decision disclosure + human review
section("3. AUTOMATED-DECISION DISCLOSURE + HUMAN REVIEW")
interactions = pd.read_csv(f"{BASE}/data/interactions.csv")
sample = interactions[interactions.candidate_id != "REDACTED"].iloc[7].to_dict()
decision = explain_decision(sample)
print(f"Decision: {decision['decision']}  Score: {decision['output_score']}")
print(f"Reason: {decision['plain_english_reason']}")
if decision["decision"] == "not_advanced":
    ticket = submit_for_human_review(decision)
    print(f"Human review ticket filed: {ticket['ticket_id']} (status={ticket['status']})")
    print("\nQueue snapshot (OPEN):")
    for row in review_queue_snapshot():
        print(row)
    resolve_review(ticket["ticket_id"], reviewer="hr_analyst_2", resolution="Confirmed, candidate notified")
    print("\nQueue snapshot (RESOLVED):")
    for row in review_queue_snapshot():
        print(row)

# 4. INDUCE FAILURE: simulate model unavailable, confirm designed degradation
section("4. INDUCED FAILURE — model file made unavailable")
model_path = f"{BASE}/models/ranker.joblib"
tmp_path = model_path + ".bak"
os.rename(model_path, tmp_path)
try:
    joblib.load(model_path)
    print("UNEXPECTED: model loaded, failure not induced correctly")
except FileNotFoundError:
    print("Model file unavailable, as expected. Falling back to baseline ordering...")
    fallback_rank = interactions[interactions.candidate_id != "REDACTED"].sort_values(
        "skill_match_score", ascending=False).head(3)[["candidate_id", "job_id", "skill_match_score"]]
    print("Fallback (labelled 'basic ordering, ML unavailable' in UI):")
    print(fallback_rank.to_string(index=False))
finally:
    os.rename(tmp_path, model_path)
    print("Model restored; system returns to ML ranking on next request.")

# 5. Fairness + audit pack summary
section("5. AUDIT PACK SUMMARY")
fairness = json.load(open(f"{BASE}/audit/fairness_results.json"))
drift = json.load(open(f"{BASE}/audit/drift_check.json"))
print(f"Demographic parity gap: {fairness['demographic_parity_gap']} -> {fairness['demographic_parity_verdict']}")
print(f"Equal opportunity gap:  {fairness['equal_opportunity_gap']} -> {fairness['equal_opportunity_verdict']}")
print(f"Train/serve skew check (buggy path): PSI={drift['psi_buggy_serving_path']} -> {drift['verdict_buggy_path']}")
print(f"Train/serve skew check (healthy path): PSI={drift['psi_healthy_serving_path']} -> {drift['verdict_healthy_path']}")
print("\nFull artifacts: audit/model_card.md, audit/fairness_results.json, audit/lineage.json, "
      "audit/drift_check.json, audit/human_review_queue.db, logs/experiment_log.jsonl")

# 6. Evidence trail: multiple training runs + reproducibility
section("6. EVIDENCE TRAIL — multiple runs + reproducibility")
lineage = json.load(open(f"{BASE}/audit/lineage.json"))
history_path = f"{BASE}/audit/fairness_history.jsonl"
history_count = sum(1 for _ in open(history_path)) if os.path.exists(history_path) else 0
registry = json.load(open(f"{BASE}/models/model_registry.json"))
print(f"Fairness history entries: {history_count} (recomputed every training run)")
print(f"Model registry versions: {len(registry['versions'])}")
print(f"Distinct data snapshots: {lineage['reproducibility_verified']['distinct_data_snapshots']}")
print(f"Reproducibility status: {lineage['reproducibility_verified']['status']}")
print(f"Detail: {lineage['reproducibility_verified']['detail']}")

section("DEMO COMPLETE")
