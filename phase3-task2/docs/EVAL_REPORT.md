# Evaluation Report — PlaceMux Candidate-Job Ranking Model

Generated: 2026-07-19T15:07:38.026539+00:00
Model version evaluated: `model_v20260719_150716`
Feature pipeline version: `fp_v1.0.0`

## Held-out set
- 15,000 impression rows, days 24-29 (never used in training or tuning)
- Split strategy: time-based (train day_idx<24, test day_idx>=24)

## Offline metrics (held-out, never tuned on)
| Metric | Model | Baseline (popularity) | Gap |
|---|---|---|---|
| nDCG@10 | 0.8007 | 0.7340 | +0.0667 |
| Precision@5 | 0.3999 | 0.3930 | +0.0069 |

Quality-floor SLO (nDCG@10 >= 0.6): **PASS** (0.8007)

## Simulated online-effect proxy (top-1-shown CTR)
- Model: 0.5004 | Baseline: 0.4040 | Uplift: +23.9%

**Ship gate**: both the offline nDCG win AND the online CTR-proxy win must be positive. Offline gap positive, online-proxy gap positive -> SHIP.

## Feature importances (used for per-prediction explanations at serve time)
- skill_overlap_ratio: 0.4413
- exp_gap: 0.3332
- salary_gap_abs: 0.1487
- region_match: 0.0372
- job_popularity_log: 0.0125
- cand_activity_score: 0.027

## Known gap / rejected alternative
Objective is pointwise regression on graded relevance (0-3), not full pairwise/listwise LambdaMART, because no network egress was available in this environment to install LightGBM/XGBoost. `src/training/train_model.py:build_model()` isolates the model class behind one function so a LambdaMART objective can be swapped in without touching feature code, evaluation code, or serving code.