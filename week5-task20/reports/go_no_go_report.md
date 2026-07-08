# PlaceMux Rec Validation — Go/No-Go Report

**Verdict: GO**

## Decision Criteria

| Criterion | Threshold | Current | Met? |
|---|---|---|---|
| Model beats baseline | Fresh P@5 ≥ Baseline P@5 | 0.5003 vs 0.4784 | ✅ |
| Drift acceptable | Drift AUC < 0.75 | 0.5554 | ✅ |
| Dry run passed | All steps pass | 15/15 | ✅ |

## Out-of-Sample Validation

| Metric | Original Model | Fresh Baseline | Fresh Model | Δ vs Baseline | Δ vs Original |
|---|---|---|---|---|---|
| Precision@5 | 0.4741 | 0.4784 | 0.5003 | +0.0219 | +0.0262 |
| Recall@5 | 0.7943 | 0.6722 | 0.7016 | | |
| MRR | 0.7377 | 0.7052 | 0.7239 | | |
| FPR@5 | 0.6264 | 0.5987 | 0.5803 | | |

## Drift Detection

- **AUC**: 0.5554
- **Severity**: minor
- **Interpretation**: Minor distribution drift detected (AUC=0.5554). The classifier can weakly distinguish old from new data. This is expected when scaling up; model performance should be monitored but is likely still valid.
- **Drifted features (KS-test p < 0.05)**: match_score, skill_gap_count, verified_skill_count, ai_trust_score, skill_gap_ratio, trust_weighted_score, college_avg_match_score

## Dry Run Summary

- **Total steps**: 15
- **Passed**: 15
- **Failed**: 0
- **Isolation checks**: 3/3
- **Deliberate failures handled**: 3/3
