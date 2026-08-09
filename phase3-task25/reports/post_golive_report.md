# Post-Go-Live Model-Health Report & Forward Roadmap
Task 25 / Sprint E

## Rollout monitoring (evidence: reports/rollout_monitor_log.csv, reports/rollback_decision.json)
Rollback trigger metric: **PSI(exp_years) vs pre-rollout training distribution**, threshold 0.25.
During the simulated day>=20 rollout (5% -> 25% -> 100% traffic) the trigger
**fired on day 20**.
Root cause: feature 'exp_years' under-logged by serving pipeline starting day 20 (simulated real bug).
Action taken: freeze traffic ramp at current stage, page on-call, revert serving to previous model version within 5 min.
This is the exact "shipping an offline win that never gets validated online"
failure mode (Sec 12) being caught live, not after the fact.

## Model health summary
- Offline quality holds vs. baseline (see certification_pack.md, Sec 1).
- Fairness gates hold (Sec 2).
- A real serving-side data-quality regression was detected and would have
  been rolled back automatically before it reached 100% of traffic.

## Answers to the brainstorming questions (Sec 9)
**What is the one number that tells you to roll back?**
PSI(exp_years) vs. the pre-rollout training distribution, threshold 0.25.
It is the single feature most exposed to train/serve skew in this pipeline.

**What did you knowingly leave unfixed, and who owns it?**
Precision@10 parity with the baseline (no lift) is unresolved - ranking
order improved, raw relevant-item recall did not. Owned by AI/ML Engineering
for Phase 4; needs a listwise objective retune, not a quick fix.
The online proxy (IPS) has high variance on 0.9961 propensity
clipping and should be replaced by a real 5% A/B slice in week 1 of go-live -
owned by the same team, tracked in the rollout monitor.

**What is the first thing you would improve in Phase 4?**
Close the precision gap with a listwise loss (LambdaMART -> full listwise
NDCG objective) and replace the exp_years field with a serving-side schema
contract + validation, so the class of bug caught here in monitoring is
caught at write-time instead.

## Forward roadmap (Phase 4)
1. Replace IPS proxy with real online A/B once 100% rollout is stable for 2 weeks.
2. Add a schema-validation gate on `exp_years` at the feature-store write path.
3. Retrain with a full listwise ranking loss to close the Precision@10 gap.
4. Expand fairness audit from 2 groups to all DPDP-relevant protected classes.
5. Move drift monitoring from daily batch (this report) to streaming (sub-hour).

## Hand-off (Sec 13)
Certified model, registry, rollout monitor, and rollback runbook are handed
to Production Ownership. Runbook: on PSI(exp_years) > 0.25, revert to the
previous registry entry in `registry/model_registry.json` and page on-call;
no manual investigation required to *stop the bleeding*, only to fix root cause.
