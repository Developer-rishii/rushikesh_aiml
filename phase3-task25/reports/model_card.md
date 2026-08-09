# Model Card - PlaceMux Ranking Model v2.0

**Trained:** Sun Aug  9 19:16:35 2026
**Owner:** AI/ML Engineering, Sprint E
**Training data:** 39926 logged rows (day<20), held out 20074 rows (day>=20)
**Data split:** time-based day<20 train / day>=20 held-out
**Features:** skill_score, exp_years, job_seniority, job_comp_level, fit_gap
**Objective:** LambdaMART (lambdarank), NDCG@10

## Intended use
Ranks job postings for a candidate at impression time. NOT used to make
autonomous hire/reject decisions - always shown to a human recruiter/candidate.

## Known limitations
- `exp_years` field is vulnerable to a serving-side under-logging bug
  (caught by drift_rollback.py's PSI monitor, see rollback_decision.json).
- Precision@10 lift over baseline is ~0 (see offline_eval.json) - the win is
  concentrated in ranking order (nDCG/MAP), not in raw relevant-item recall.
  Flagged, not hidden.

## Fairness
Demographic parity gap: 0.0096 (threshold 0.08, PASS)
Equal opportunity gap: 0.0037 (threshold 0.08, PASS)

## Serving
p95 latency: 0.435 ms (SLO 150 ms - MET)
Estimated cost: $1.3e-05 / 1000 requests

## Versioning & rollback
Model artifact: `D:\Placemux-aiml\phase3-task25/registry/models/ranker_v2.0.pkl`
Registry: `registry/model_registry.json` (append-only, every training run logged)
Rollback trigger: PSI(exp_years) > 0.25 vs training distribution -> revert to previous version within 5 min.
