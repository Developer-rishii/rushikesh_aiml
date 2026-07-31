
======================================================================
STEP: 1/5 Generate realistic tenant logs (idempotent, seeded)
======================================================================
tenant A logs: (12000, 26), tenant B logs: (9000, 26)
Base rates -- A applied: 0.6650833333333334 B applied: 0.6606666666666666

======================================================================
STEP: 2/5 Train per-tenant models + evaluate vs baseline (evidence/metrics_report.json)
======================================================================
{
  "tenant_id": "tenantA",
  "version": "20260731T161604Z",
  "n_train": 9000,
  "n_test": 3000,
  "model_auc": 0.8109,
  "baseline_auc": 0.5,
  "model_ndcg@10": 1.0,
  "baseline_ndcg@10": 0.662,
  "model_precision@5": 1.0,
  "baseline_precision@5": 0.6,
  "gap_offline_expected_online": "Offline nDCG uplift is measured on logged/held-out impressions only; expected online effect will be SMALLER due to position bias and exploration deficit in logged data. Recommend online A/B before full rollout, ramping 5% -> 25% -> 100%."
}
{
  "tenant_id": "tenantB",
  "version": "20260731T161606Z",
  "n_train": 6750,
  "n_test": 2250,
  "model_auc": 0.7792,
  "baseline_auc": 0.5,
  "model_ndcg@10": 1.0,
  "baseline_ndcg@10": 0.7386,
  "model_precision@5": 1.0,
  "baseline_precision@5": 0.6,
  "gap_offline_expected_online": "Offline nDCG uplift is measured on logged/held-out impressions only; expected online effect will be SMALLER due to position bias and exploration deficit in logged data. Recommend online A/B before full rollout, ramping 5% -> 25% -> 100%."
}

Wrote evidence/metrics_report.json and experiments/experiment_log.md

======================================================================
STEP: 3/5 Serve both tenants + induce model-unavailable failure
======================================================================

[tenantA] serving mode=model v20260731T161604Z threshold=0.55 cap=20
candidate_id      job_id    score  shortlisted
tenantA_c198 tenantA_j49 0.959740         True
 tenantA_c56 tenantA_j35 0.938569         True
tenantA_c613 tenantA_j35 0.879396         True
tenantA_c436 tenantA_j58 0.868735         True
tenantA_c637  tenantA_j2 0.861650         True

[tenantB] serving mode=model v20260731T161606Z threshold=0.65 cap=15
candidate_id      job_id    score  shortlisted
tenantB_c219 tenantB_j22 0.948073         True
tenantB_c522  tenantB_j5 0.943726         True
tenantB_c117  tenantB_j4 0.903858         True
tenantB_c128  tenantB_j5 0.900818         True
tenantB_c152  tenantB_j8 0.740439         True

--- Failure scenario: tenantA model file goes missing/unavailable ---
[tenantA DEGRADED] serving mode=global_popularity_baseline
candidate_id      job_id  score  shortlisted
tenantA_c179 tenantA_j48    0.5        False
tenantA_c416 tenantA_j31    0.5        False
tenantA_c436 tenantA_j58    0.5        False
 tenantA_c98  tenantA_j3    0.5        False
tenantA_c637  tenantA_j2    0.5        False

======================================================================
STEP: 4/5 Prove isolation: access-control + cross-serving + leakage probe
======================================================================
=== Test 1: access-control (can tenantA's store ever return tenantB rows?) ===
Rows returned by tenantA store belonging to another tenant: 0 (expect 0)
PASS: unknown tenant id correctly rejected by TenantDataStore

=== Test 2: cross-serving (score tenantB rows through tenantA's service) ===
PASS: cross-tenant scoring blocked -> ISOLATION BREACH: attempted to score rows from another tenant

=== Test 3: membership-inference leakage probe on tenantA's model ===
Membership-inference attacker AUC distinguishing tenantA-trained-on rows vs tenantB rows never seen in training: 0.546
Interpretation: an AUC far from 0.5 would mean the model's confidence pattern reveals which tenant's data trained it (a leakage signature); note this number is expected to be >0.5 here simply because tenantA's model is legitimately MORE ACCURATE on tenantA's own skill-weighting distribution (that's the product working as intended, not leakage). The leakage-specific check is Test 1 + Test 2: tenantB's raw rows are structurally unreachable by tenantA's store/service, so no tenantB row can ever appear in a tenantA training call in the first place -- verified by the assertion in TenantDataStore.load_logs().

Wrote /home/claude/task16/evidence/isolation_proof.txt

======================================================================
STEP: 5/5 Fairness audit + drift monitor (with simulated shift alert)
======================================================================
{
  "tenant_id": "tenantA",
  "shortlist_rate_group_A": 0.6985,
  "shortlist_rate_group_B": 0.6835,
  "demographic_parity_gap": 0.015,
  "flag": "OK"
}
{
  "tenant_id": "tenantB",
  "shortlist_rate_group_A": 0.5694,
  "shortlist_rate_group_B": 0.5676,
  "demographic_parity_gap": 0.0017,
  "flag": "OK"
}
{"tenant_id": "tenantA", "psi": 0.0, "flag": "OK"}
{"tenant_id": "tenantB", "psi": 0.0, "flag": "OK"}
{"tenant_id": "tenantA", "psi": 2.8772, "flag": "ALERT"}

DONE. Evidence written to evidence/*.json and evidence/isolation_proof.txt
Experiment log: experiments/experiment_log.md
