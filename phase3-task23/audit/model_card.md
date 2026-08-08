# Model Card — PlaceMux Candidate Ranking Model v1.0.0

## Intended use
Ranks candidates against job postings to support (not replace) recruiter shortlisting.
NOT intended as a sole/final hiring decision — output always routes through a human-review
path for rejections (see disclosure.py), per DPDP/GDPR Art.22.

## Model type & training
- Type: lightgbm_lambdarank
- Trained: 2026-08-08T13:52:42.466460+00:00
- Training dataset hash: 04d06a92e692f4af
- Model artifact hash: 6fc13f505de5e915
- Features used: years_experience, skill_match_score, profile_completeness, seniority_level, req_skill_score, recency_feature_train
- Explicitly excluded: protected_group (fairness-only, never a model input)

## Offline performance (held-out, not tuned on)
- nDCG@10: 0.0298 vs baseline 0.0124
  (140.32% lift)
- MAP: 0.0274 vs baseline 0.0143
- Precision@10: 0.0105 vs baseline 0.0053
- Evaluated on 38 held-out jobs / 15286 rows

## Online effect — HONEST CAVEAT
No live A/B test was run; this is a held-out behavioural proxy from logged data, NOT a claim of validated online lift. Real online validation is an explicit dependency handed off in Stage E / Section 13.
Proxy apply-rate (top-10): model 0.0263 vs
baseline 0.0184. This is NOT a validated
live result and must not be reported as one.

## Fairness (recomputed every training run)
- Demographic parity gap: 0.0008 — PASS (<0.10)
- Equal opportunity gap: 0.0022 — PASS (<0.10)
- Protected attribute: protected_group (synthetic, for audit only — never used as a model feature)

## Known limitations
- Trained on synthetic-but-realistic logged data (no access to PlaceMux production DB in
  this study-guide context) — see data/data_manifest.json for full provenance disclosure.
- Data generation is calibrated to industry benchmarks (~8% CTR, ~20% apply rate, ~15% shortlist rate)
  and includes a messy validation slice to test robustness, but remains fundamentally synthetic.
- Train/serve skew was deliberately tested and IS caught by drift_monitor.py (see audit/drift_check.json).
- Non-linear model means individual-row influence is not exactly subtractable; deletion is
  handled via retention-window purge + scheduled retrain, not per-request retraining (see
  Design Decision in dsr_rights.py).

## Human oversight
Every "not_advanced" automated decision is disclosed with a plain-English reason and
auto-filed to a human-review queue (audit/human_review_queue.db). Fallback if model is
unavailable: chronological + skill-match baseline ordering, clearly labelled in UI.

## Data subject rights
- Access (Art.15): dsr_rights.access_request()
- Erasure (Art.17): dsr_rights.deletion_request() — executes real, on-disk deletion + pseudonymisation
