# Model Card — PlaceMux Candidate Ranker v2

## Model details
- **Version:** v2
- **Parent version:** v1
- **Type:** LightGBM LambdaMART (listwise learning-to-rank), grouped by job_id
- **Training data hash:** `b4e5e83f6d52c6ca`
- **Feature schema hash:** `684c7dfe542ba072`
- **Trained on:** 15400 logged impression rows
- **Registered:** 2026-07-29T16:12:15

## Intended use
Ranks candidates within a single job's applicant pool so recruiters see the
most likely-to-be-shortlisted candidates first. NOT intended to make
autonomous accept/reject decisions -- output is a ranking signal shown to a
human recruiter, per the DPDP constraint on automated hiring decisions
(Section 3).

## Training data
Real-style logged impressions (impression -> click -> application ->
shortlist funnel), features computed by `src/features.py` (the single
feature-computation layer shared with serving, to prevent train/serve
skew). Protected attribute (`gender`) is excluded from model features and
used only for the fairness audit below.

## Offline metrics (held-out, not tuned on)
| Metric | Baseline (skill_match_score only) | This model | Delta |
|---|---|---|---|
| nDCG@10 | 0.9393 | 0.9561 | 0.0168 |
| MAP@10 | 0.9178 | 0.9406 | 0.0228 |
| Precision@5 | 0.0339 | 0.0339 | 0.0 |

**Offline-to-online gap:** offline metrics are computed on logged,
already-ranked impressions (position bias exists). This is a KNOWN
limitation, not swept under the rug: Stage E's demo compares this
model's offline win against post-deployment shortlist-rate as the
online proxy, and any model whose offline win doesn't survive that
check is flagged, per the pitfall "shipping an offline win that never
gets validated online."

## Fairness audit (gender: F vs M)
| Metric | Value | Threshold | Pass? |
|---|---|---|---|
| Demographic parity difference | 0.0033 | < 0.10 | True |
| Equal opportunity difference | 0.0349 | < 0.10 | True |

Selection rate by group: {'F': 0.2076, 'M': 0.2109}

**Overall fairness gate: True**

## Monitoring & rollback
- Drift monitored via PSI on all 6 input features + performance drift on
  nDCG@10 (see `src/drift.py`). Thresholds: PSI alert >= 0.25, performance
  drop >= 8% relative.
- Rollback path: `ModelRegistry.rollback(version)` re-points the
  production pointer to any prior registered version in O(1), fully
  audited in the `promotions` table (who/when/why).
- Failure mode: if the production artifact is unavailable or fails to
  load, `src/serve.py` falls back to `SkillMatchBaseline` and logs a
  `degraded_mode=True` event rather than failing the request.

## Limitations
- Binary relevance label (shortlisted) only; does not yet use graded
  relevance (click < applied < shortlisted), which would give the ranker
  more signal per impression -- deferred, tracked as future work.
- Trained on 15400 rows from a single 180-day simulated window;
  seasonal effects beyond that window are unvalidated.
- Fairness audit covers only the `gender` protected attribute recorded in
  this dataset; region-level fairness was not separately gated in v1.

## Who to contact
Governance artifacts hand off to Compliance/DevOps per Section 13; this
card plus the registry DB (`governance/registry.db`) is the complete
audit trail for "which model made a decision on a given date."
