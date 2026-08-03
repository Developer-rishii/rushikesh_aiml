# Experiment Log

## Run 1 — data generation
Command: `python3 data_gen.py`
Seed: 42 | candidates=4000, jobs=120, impressions=7200
Per-tenant funnel rates (click/apply/shortlist), all ~within 1pt of each
other by design (tenants differ in hiring bar via policy config, not raw
data generation) — see `evidence/01_data_gen.log`.

## Run 2 — model training + offline eval
Command: `python3 model.py`
Split: GroupShuffleSplit by job_id, test_size=0.2, seed=42 (24 held-out jobs)
Model: GradientBoostingRegressor(n_estimators=150, max_depth=3, lr=0.08, seed=42)
Baseline: skill_overlap only (proxy for pre-ML rules-based matching)

| Metric | Model | Baseline | Lift |
|---|---|---|---|
| nDCG@10 | 0.9551 | 0.9032 | +5.7% |
| precision@10 | 0.1583 | 0.1458 | +8.6% |
| MAP | 0.2405 | 0.2332 | +3.1% |

Full output: `evidence/02_model_train_eval.log`

## Run 3 — Stage E integration + break-it demo
Command: `python3 demo.py`
- Step 1: baseline live ranking for job J0031 / tenant acme_bank, config v1
- Step 2: admin previews config v2 (w_skill 0.5→0.7, max_distance 500→60km)
  — guardrail passed, fairness rates near-parity (A: 0.163, B: 0.171),
    funnel impact −1.2% eligible, shown BEFORE commit
- Step 3: committed v2, live re-rank confirmed changed (top-2 scores moved
  0.885→0.905 / 0.837→0.865); `ranking actually changed = True`
- Step 4: adversarial config (w_distance=3.0, min_skill_overlap=0.95)
  rejected on 4 independent structural grounds; live version confirmed
  unchanged after rejection
- Step 5: `simulate_model_down=True` → `degraded_mode=True`, service kept
  returning ranked results via rule-only fallback instead of failing
- Step 6: fairness audit on real (mildly biased-by-construction) logs for
  orion_retail passed (rates 0.164 vs 0.170, ratio > 0.8)

Full output: `evidence/03_full_demo.log`

## Run 4 — fairness guardrail active-rejection proof
Command: inline script, see `evidence/04_fairness_guardrail_trigger.log`
Amplified a distance-based proxy for the protected attribute in the audit
sample (legacy-rule scenario) and proposed a distance-heavy config →
guardrail computed selection rates A=0.312, B=0.000, ratio=0.00 < 0.8 →
**REJECTED**. Confirms the guardrail doesn't just pass on easy cases, it
actively blocks on a realistic adversarial one.
