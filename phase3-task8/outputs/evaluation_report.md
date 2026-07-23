# Honest Evaluation Report (held-out data, never trained/tuned on)

Holdout snapshots: Nov 2025 - Jan 2026. n=19075, base churn rate=0.235

## PR-AUC and lift over baseline

| Model | PR-AUC | Lift@5% | Lift@10% | Lift@20% | Lift@30% |
|---|---|---|---|---|---|
| model_v1 (HistGBM) | 0.609 | 3.30x | 3.10x | 2.65x | 2.30x |
| baseline: 14-day-inactivity rule | 0.478 | 2.78x | 2.57x | 2.23x | 1.94x |
| secondary baseline: RFM rule | 0.504 | 2.77x | 2.57x | 2.30x | 2.04x |

## Operating point (top 10% riskiest flagged)

- Model: precision=0.727, recall=0.310, f1=0.435
- Baseline (14-day rule): precision=0.467, recall=0.552, f1=0.506

## Honest gap note
This is an OFFLINE evaluation on simulated logs. The offline PR-AUC and lift numbers above are not a promise of equivalent online lift -- real online effect depends on whether growth's intervention (re-engagement email/digest) actually changes behaviour once delivered, which this offline evaluation cannot measure. Recommended next step before full rollout: an online A/B test on a held-out traffic slice, comparing intervention-on-model-flagged vs intervention-on-baseline-flagged vs no-intervention control, tracking actual 21-day reactivation rate as the online ground truth.