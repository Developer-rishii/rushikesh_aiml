# Phase 3 Backlog — Matching System
Generated: 2026-07-19T18:56:20 UTC
Sprint: Phase 3 Sprint A — Scale & Reliability

## Key Metrics at Backlog Creation
| Metric | Value |
|--------|-------|
| nDCG@5 (offline) | 0.7454 |
| CTR (online) | 0.2535 |
| Online/offline gap | -0.2885 |
| Skewed features | skill_score_feature, verified_skills, skill_gap, years_exp, model_score |
| Predicted defects | 2367 |

## Backlog (7 items: 3 P0, 4 P1)

| Rank | ID | Priority | Title | Affected | Effort |
|------|----|----------|-------|----------|--------|
| 1 | B-002 | P0 | Investigate online/offline metric gap — model over-confident | 8,000 | 8d |
| 2 | B-007 | P0 | Ensure 100% prediction logging with model version + feature  | 8,000 | 3d |
| 3 | B-001 | P0 | Fix train/serve skew in features: skill_score_feature, verif | 300 | 5d |
| 4 | B-006 | P1 | Improve matching quality for college_tier=2 (lowest CTR segm | 3,353 | 10d |
| 5 | B-004 | P1 | Remediate 'false_positive' defects in recommendation ranking | 1,087 | 6d |
| 6 | B-003 | P1 | Remediate 'skew_induced' defects in recommendation ranking | 642 | 6d |
| 7 | B-005 | P1 | Remediate 'false_negative' defects in recommendation ranking | 120 | 6d |

## Item Details

### B-002 — Investigate online/offline metric gap — model over-confident
**Priority:** P0  |  **Owner:** AI/ML Engineer  |  **Effort:** 8d

**Evidence:** nDCG@5 offline = 0.7454, CTR online = 0.2535, gap = -0.2885 (model over-confident). Model scores are systematically biased vs actual user behaviour.

**Metric to move:** CTR from 0.254 toward 0.542

### B-007 — Ensure 100% prediction logging with model version + feature snapshot
**Priority:** P0  |  **Owner:** AI/ML + Backend  |  **Effort:** 3d

**Evidence:** Without complete prediction logging, debugging online/offline gaps in future sprints is impossible. Currently logging all scored pairs but feature snapshots at serving time need to be persisted separately to enable reproducible skew detection.

**Metric to move:** Prediction log completeness → 100%

### B-001 — Fix train/serve skew in features: skill_score_feature, verified_skills, skill_gap, years_exp, model_score
**Priority:** P0  |  **Owner:** AI/ML + Data Engineering  |  **Effort:** 5d

**Evidence:** KS test detects distribution shift in 5 features. Worst affected model version: v1.2 (mean_skew=0.081). Skew accounts for a portion of the 0.288 online/offline gap.

**Metric to move:** online_offline_gap → 0

### B-006 — Improve matching quality for college_tier=2 (lowest CTR segment)
**Priority:** P1  |  **Owner:** AI/ML Engineer  |  **Effort:** 10d

**Evidence:** college_tier=2 has CTR=0.2455 — the worst-performing segment. This may reflect fairness issues carried from the bias audit (Task 21/24).

**Metric to move:** Tier 2 CTR → segment average

### B-004 — Remediate 'false_positive' defects in recommendation ranking
**Priority:** P1  |  **Owner:** AI/ML Engineer  |  **Effort:** 6d

**Evidence:** 1087 'false_positive' defects detected, mean user impact = 0.476. Mean defect rank = 1070 (lower = hurts more users).

**Metric to move:** Defect count → 0 for false_positive

### B-003 — Remediate 'skew_induced' defects in recommendation ranking
**Priority:** P1  |  **Owner:** AI/ML Engineer  |  **Effort:** 6d

**Evidence:** 642 'skew_induced' defects detected, mean user impact = 0.500. Mean defect rank = 1061 (lower = hurts more users).

**Metric to move:** Defect count → 0 for skew_induced

### B-005 — Remediate 'false_negative' defects in recommendation ranking
**Priority:** P1  |  **Owner:** AI/ML Engineer  |  **Effort:** 6d

**Evidence:** 120 'false_negative' defects detected, mean user impact = 0.431. Mean defect rank = 1396 (lower = hurts more users).

**Metric to move:** Defect count → 0 for false_negative
