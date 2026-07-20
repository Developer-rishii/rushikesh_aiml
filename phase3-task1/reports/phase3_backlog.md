# Phase 3 Backlog — Matching System
Generated: 2026-07-20T06:53:05 UTC
Sprint: Phase 3 Sprint A — Scale & Reliability

## Key Metrics at Backlog Creation
| Metric | Value |
|--------|-------|
| nDCG@5 (offline) | 0.735 |
| CTR (online) | 0.2519 |
| Online/offline gap | -0.2877 |
| Skewed features | match_score_train |
| Predicted defects | 1342 |

## Backlog (7 items: 3 P0, 4 P1)

| Rank | ID | Priority | Title | Affected | Effort |
|------|----|----------|-------|----------|--------|
| 1 | B-002 | P0 | Investigate online/offline metric gap — model over-confident | 8,000 | 8d |
| 2 | B-007 | P0 | Ensure 100% prediction logging with model version + feature  | 8,000 | 3d |
| 3 | B-001 | P0 | Fix train/serve skew in features: match_score_train | 300 | 5d |
| 4 | B-006 | P1 | Improve matching quality for college_tier=2 (lowest CTR segm | 3,489 | 10d |
| 5 | B-004 | P1 | Remediate 'false_positive' defects in recommendation ranking | 813 | 6d |
| 6 | B-003 | P1 | Remediate 'skew_induced' defects in recommendation ranking | 336 | 6d |
| 7 | B-005 | P1 | Remediate 'false_negative' defects in recommendation ranking | 41 | 6d |

## Item Details

### B-002 — Investigate online/offline metric gap — model over-confident
**Priority:** P0  |  **Owner:** AI/ML Engineer  |  **Effort:** 8d

**Evidence:** nDCG@5 offline = 0.7350, CTR online = 0.2519, gap = -0.2877 (model over-confident). Model scores are systematically biased vs actual user behaviour.

**Metric to move:** CTR from 0.252 toward 0.540

### B-007 — Ensure 100% prediction logging with model version + feature snapshot
**Priority:** P0  |  **Owner:** AI/ML + Backend  |  **Effort:** 3d

**Evidence:** Without complete prediction logging, debugging online/offline gaps in future sprints is impossible. Currently logging all scored pairs but feature snapshots at serving time need to be persisted separately to enable reproducible skew detection.

**Metric to move:** Prediction log completeness -> 100%

### B-001 — Fix train/serve skew in features: match_score_train
**Priority:** P0  |  **Owner:** AI/ML + Data Engineering  |  **Effort:** 5d

**Evidence:** KS test detects distribution shift in 1 features. Worst affected model version: v1.1 (mean_skew=0.081). Skew accounts for a portion of the 0.288 online/offline gap.

**Metric to move:** online_offline_gap -> 0

### B-006 — Improve matching quality for college_tier=2 (lowest CTR segment)
**Priority:** P1  |  **Owner:** AI/ML Engineer  |  **Effort:** 10d

**Evidence:** college_tier=2 has CTR=0.2459 — the worst-performing segment. This may reflect fairness issues carried from the bias audit (Task 21/24).

**Metric to move:** Tier 2 CTR -> segment average

### B-004 — Remediate 'false_positive' defects in recommendation ranking
**Priority:** P1  |  **Owner:** AI/ML Engineer  |  **Effort:** 6d

**Evidence:** 813 'false_positive' defects detected, mean user impact = 0.430. Mean defect rank = 706 (lower = hurts more users).

**Metric to move:** Defect count -> 0 for false_positive

### B-003 — Remediate 'skew_induced' defects in recommendation ranking
**Priority:** P1  |  **Owner:** AI/ML Engineer  |  **Effort:** 6d

**Evidence:** 336 'skew_induced' defects detected, mean user impact = 0.486. Mean defect rank = 581 (lower = hurts more users).

**Metric to move:** Defect count -> 0 for skew_induced

### B-005 — Remediate 'false_negative' defects in recommendation ranking
**Priority:** P1  |  **Owner:** AI/ML Engineer  |  **Effort:** 6d

**Evidence:** 41 'false_negative' defects detected, mean user impact = 0.382. Mean defect rank = 862 (lower = hurts more users).

**Metric to move:** Defect count -> 0 for false_negative
