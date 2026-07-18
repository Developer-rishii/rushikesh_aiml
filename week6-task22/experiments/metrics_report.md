# PlaceMux -- Drift + Retraining: Metrics Report

## Baseline vs Trained Model (held-out month 2026-03)

| Metric | Baseline (overlap-ratio) | Trained model | Lift |
|---|---|---|---|
| Precision | 0.3239 | 0.4441 | +0.1202 |
| Recall | 0.7404 | 0.41 | -0.3304 |
| False Positive Rate | 0.4513 | 0.1499 | +0.3014 (reduction) |
| ROC AUC | 0.7083 | 0.7707 | - |

## Segment Breakdown (held-out month, trained model)

- **meets_or_exceeds** (n=927): precision=0.4384, recall=0.4723, fpr=0.25
- **underqualified** (n=573): precision=0.5238, recall=0.1618, fpr=0.0198

## Drift Monitoring & Retraining Timeline

| Month | Drift status | Max feature PSI | Pred PSI | Event | Model | Precision | Recall | FPR |
|---|---|---|---|---|---|---|---|---|
| 2026-01+2026-02+2026-03 | - | - | - | initial_train | v1 | 0.641 | 0.3266 | 0.0553 |
| 2026-04 | MODERATE_DRIFT | 0.2462 | 0.0043 | monitor_only | v1 | 0.4379 | 0.2087 | 0.0729 |
| 2026-05 | MODERATE_DRIFT | 0.2487 | 0.0179 | monitor_only | v1 | 0.4601 | 0.2259 | 0.0753 |
| 2026-06 | SIGNIFICANT_DRIFT | 0.4753 | 0.0096 | drift_triggered_retrain | v2 | 0.6489 | 0.5683 | 0.084 |

## Final Model: `v2`

### Feature importance

- weighted_skill_score: 0.6616
- years_gap: 0.1947
- overlap_ratio: 0.0585
- student_breadth: 0.0436
- overlap_count: 0.0227
- jd_breadth: 0.0125
- missing_top_skill: 0.0064

### Regression check across all 6 months

| Month | Precision | Recall | FPR | ROC AUC | n |
|---|---|---|---|---|---|
| 2026-01 | 0.45 | 0.3709 | 0.1452 | 0.7303 | 1500 |
| 2026-02 | 0.4844 | 0.4545 | 0.1424 | 0.7689 | 1500 |
| 2026-03 | 0.439 | 0.4454 | 0.1662 | 0.7557 | 1500 |
| 2026-04 | 0.4027 | 0.3676 | 0.1484 | 0.7372 | 1500 |
| 2026-05 | 0.4169 | 0.4006 | 0.1592 | 0.7435 | 1500 |
| 2026-06 | 0.6489 | 0.5683 | 0.084 | 0.8763 | 1500 |
