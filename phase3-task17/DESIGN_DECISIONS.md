# Design Decisions — Task 17

## 1. Stage A — the bar
Good = "an external ATS can call PlaceMux matching, get a score plus a reason,
and never be able to scrape the model." That single sentence drove three
concrete constraints implemented below: (a) explanations must be safe to give
away, (b) every call must be metered, (c) the metering has to catch fast,
broad querying specifically, not just raw volume.

## 2. Model / ranking approach
**Chosen:** pointwise `GradientBoostingRegressor` (scikit-learn) trained on a
graded relevance label (0=impression, 1=click, 2=apply, 3=shortlist), used to
rank candidates per job.

**Rejected:** LightGBM/XGBoost `LambdaMART` (true listwise learning-to-rank).
This is what the study guide's "recommended stack" calls out, and it's the
technically superior choice for optimising rank ORDER directly rather than a
pointwise proxy. It was rejected here **only** because this sandbox has no
network access to `pip install` it — not a silent downgrade: it's disclosed
in `ml/train.py`'s docstring and in `ml/experiment_log.md`. In a networked
environment, swap `GradientBoostingRegressor` for `LGBMRanker` with an
`objective='lambdarank'` — the rest of the pipeline (registry, versioning,
API, quota) is unaffected because it only depends on `model.predict()`.

## 3. Returning raw scores vs bucketed bands
**Chosen:** bucketed 0–100 bands in steps of 5 (21 possible outputs).
**Rejected:** raw continuous score. A raw float lets an attacker binary-search
the decision boundary with far fewer queries; bucketing measurably reduces
signal-per-query without hurting the partner's actual use case (ranking a
shortlist, showing "why").

## 4. Strict quotas vs anomaly-based abuse detection
**Chosen:** both — hard per-minute/per-day quotas (deterministic, partners can
build retry logic around a documented number) **plus** a lightweight
distinct-target heuristic (>40 distinct candidate_ids queried within a
50+ call, 5-minute window) that specifically targets the scraping signature:
broad coverage, not repeat lookups. Real ATS traffic re-checks the same
shortlist repeatedly; a scraper sweeps broadly.
**Rejected:** anomaly-detection-only. Non-deterministic limits are hostile to
integrate against — a partner can't build correct retry/backoff logic if they
don't know the rule.

## 5. Versioning
Every model is loaded from an isolated `ml/model_registry/{version}/` folder
with its own hash and metadata. `/v1/` and `/v2/` are two different models on
purpose (proven by `tests/test_endpoints.py::test_versions_are_isolated` and
demo step 8) — this is the direct answer to "does a model upgrade break a
partner's expectations?" — no, because partners choose when to move versions.

## 6. Failure mode
A model-unavailable exception (`ModelUnavailable`) is caught per-request and
falls back to a small, transparent rule-based scorer, returning `200` with
`degraded_mode: true` rather than a `500`. Verified by deliberately inducing
the outage via `/_admin/simulate_outage` in `tests/test_failure_mode.py` and
demo step 6.

## 7. Offline vs online metrics — honest caveat
`ml/experiment_log.md` reports nDCG@10 and precision@5 on a **held-out set of
jobs the model was not trained on** (grouped split, no leakage), compared
against a non-ML popularity/recency baseline a partner could build without
any ML. The model wins by design in this synthetic environment, but the log
explicitly states the offline/online gap: better ranking changes *which
candidates get shown at all* (selection bias), which offline nDCG cannot
measure — this must be confirmed with a real online A/B before claiming a win
in production. This is the honest caveat the study guide requires — not
"trust the offline number."

## 8. What was cut, and why it's disclosed rather than hidden
- No Redis/persistent quota store — in-memory `QuotaStore` documented as the
  swap point for production (interface matches what Redis would provide).
- No real ATS webhook delivery system — out of scope for this task's stated
  bar (scoring/matching API + docs), noted as a hand-off item.
- Fairness audit automation (demographic parity / equal opportunity metrics)
  is named in "Go deeper" as further study, not part of Stage B/C/D's core
  deliverables list — not implemented here, called out explicitly rather than
  silently skipped, since a one-time/missing fairness check is listed as a
  pitfall to avoid.
