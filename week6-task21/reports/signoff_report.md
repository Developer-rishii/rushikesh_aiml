# Task 21 — Fairness Audit (Start)
**AI/ML Engineer deliverable · Week 6 Phase 3 · PlaceMux · Altrodav Technologies**

Run timestamp: `2026-07-18 16:04 UTC`

## Verdict
```
Status:                   ⚠️ BIAS DETECTED — AUDIT IN PROGRESS
college_tier disparate_impact: 0.0100  ❌ FAILS 4/5ths rule (<0.80)
region disparate_impact:       0.5855  ❌ FAILS 4/5ths rule
Callable live:            GET /audit/report
```

## 1 · What's in here
This is the fairness audit for PlaceMux's recommendation engine. The team-wide theme is DPDP Consent & Security Foundations — data deletion, consent flows, and load testing are owned by other roles. This AI/ML slice audits whether the recommendation model treats students from different college tiers and regions equitably, measures disparate impact, and trains an ML bias classifier to flag individual biased outcomes to admins.

**Upstream dependency:** 'Sufficient data' — integrated recommendations from prior matching tasks. Validated at load time.

## 2 · Baseline fairness metrics (rule-based, no ML)

### By college_tier
| Tier | N pairs | Rec rate | Skill rec rate | Recall | Mean match score |
|------|---------|----------|----------------|--------|-----------------|
| 1 | 3630 | 0.3705 | 0.4096 | 0.9045 | 0.5733 |
| 2 | 4860 | 0.1481 | 0.1652 | 0.8966 | 0.4536 |
| 3 | 3510 | 0.0037 | 0.0601 | 0.0616 | 0.1806 |

**Disparate impact (college_tier):** 0.0100  ❌ FAILS 4/5ths rule
**Max parity gap:** 0.3668
**Equal opportunity gap:** 0.8429

### By region
| Region | N pairs | Rec rate | Skill rec rate | Recall | Mean match score |
|--------|---------|----------|----------------|--------|-----------------|
| rural | 2370 | 0.1198 | 0.2342 | 0.5117 | 0.3647 |
| semi_urban | 3540 | 0.1548 | 0.1672 | 0.9257 | 0.3930 |
| urban | 6090 | 0.2046 | 0.2223 | 0.9202 | 0.4374 |

**Disparate impact (region):** 0.5855  ❌ FAILS 4/5ths rule

## 3 · ML bias classifier

| Metric | Value |
|--------|-------|
| Labeled pairs (reviewed) | 2380 |
| Biased in labels | 84 (3.5%) |
| Threshold | 0.3 |
| Precision | 1.0 |
| Recall | 1.0 |
| F1 | 1.0 |
| FPR | 0.0 |

### Segment by college_tier
| Tier | N | Precision | Recall | F1 |
|------|---|-----------|--------|----|
| tier_1 | 711 | 1.0 | 1.0 | 1.0 |
| tier_2 | 945 | 1.0 | 1.0 | 1.0 |
| tier_3 | 724 | 1.0 | 1.0 | 1.0 |

### Segment by region
| Region | N | Precision | Recall | F1 |
|--------|---|-----------|--------|----|
| rural | 471 | 1.0 | 1.0 | 1.0 |
| semi_urban | 684 | 1.0 | 1.0 | 1.0 |
| urban | 1225 | 1.0 | 1.0 | 1.0 |

## 4 · Edge cases tested
| Case | Test name | Status |
|------|-----------|--------|
| Malformed input (missing columns) | test_validate_input_missing_cols | ✅ |
| Empty dataframe | test_validate_input_empty | ✅ |
| Single-group data (no disparity possible) | test_single_group_no_crash | ✅ |
| Unknown student at inference | test_predict_unknown_student | ✅ |
| Perfect disparity (DI=0.0) detection | test_perfect_bias_detected | ✅ |

## 5 · Scope note
DPDP user data deletion, consent flows, and load testing are owned by the backend/security role this week. This AI/ML slice covers only the fairness measurement layer. The self-check questions about data deletion and load testing do not apply to this deliverable.

## 6 · Hand-off
Hand-off: **Bias findings** — the fairness report JSON at `reports/fairness_report.json` and the live `/audit/report` endpoint. Guardrail: re-run this audit after every 1,000 new students onboarded; alert if disparate_impact on college_tier drops below 0.80.

## 7 · What is still open before launch (AI/ML slice only)
- Replace synthetic data with real production recommendation outputs
- Expand admin-reviewed label set (currently 2380 pairs — needs ≥500 for production-grade classifier confidence)
- Wire audit to re-run automatically on schedule