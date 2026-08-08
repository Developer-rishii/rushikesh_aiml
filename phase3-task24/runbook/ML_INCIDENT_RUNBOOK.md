# ML Incident Runbook — PlaceMux Matching Service

**Scope:** matching/ranking model serving. Trigger: a page fires from `Alerting`
(severity `warning` or `critical`) or matching quality visibly degrades
(applications drop, rankings look wrong).

## 0. Is a stale model worse than no model here?
No. A stale-but-recently-good model is safer than an error page, but staler
than `max_age_days` (2 days) is treated as equivalent to "no model" and the
service degrades to the heuristic — see `feature_store.py`. The heuristic is
always safe because it never depends on model freshness.

## 1. First action at 3am (0–5 minutes)
1. Check `evidence/alerts.log` (prod: PagerDuty context) for the `reason` field:
   - `model_service_down` → go to §2
   - `stale_features` → go to §3
   - `corrupted_or_invalid_features` → go to §4
2. Confirm the system is **degraded, not down**: query `/health/matching` (or
   run `MatchingService.match_score(...)` locally) — you should get a score
   back with `mode="heuristic"`. If you get an exception or no response at
   all, this is a bigger outage than the ML layer — escalate to platform on-call.
3. Post in `#incident-ml` with: reason, mode, time started, blast radius
   (all traffic, or one region/segment).

## 2. Model service down
- Root cause candidates: bad deploy, OOM, GPU/CPU quota, upstream (feature
  store / vector DB) timeout cascading into the model pod.
- Check model service logs / pod status. Roll back to last known-good model
  version (see model registry — every model is versioned, so you can always
  say which model produced a decision).
- While rolled back or restarting: **do nothing** — the heuristic is already
  serving traffic automatically. There is no manual failover step required;
  that itself is worth confirming (`mode == "heuristic"` in recent logs).
- Once healthy, confirm automatic recovery: next request should show
  `mode="model"` with no code change or restart of the matching service.

## 3. Stale features
- Check the feature-computation pipeline / feature store job for failures
  (cron didn't run, upstream table didn't land, backfill in progress).
- Staleness only affects the specific (candidate, job) pairs whose snapshot
  is old — check `age_days` in the alert context to scope blast radius.
- Fix the freshness job; do NOT manually bump `max_age_days` to silence the
  alert — that reintroduces the silent-failure mode this system exists to
  prevent.

## 4. Corrupted / invalid features
- This is the most serious case: NaNs, out-of-range values, or a schema
  change reaching serving. The `FeatureStore.validate()` gate should catch
  this and route to heuristic automatically — confirm it did
  (`no_nan_crash: true` in chaos evidence).
- Identify the corrupting upstream job (feature pipeline deploy, schema
  migration, bad partition). Freeze that pipeline's writes until fixed.
- **What we would not currently catch:** a corruption that produces
  *plausible but wrong* values (e.g., skill_match stuck at exactly 0.5 for
  every row) — the range/NaN validator passes it. Mitigation: add a
  distribution-drift check (compare live feature histograms to a
  reference window) as a follow-up, not yet implemented.

## 5. Fairness / DPDP note
Any period where the heuristic served real traffic must be flagged to the
fairness audit owner — the heuristic has not been through the same
demographic-parity check as the ML model and should not silently substitute
for it over long windows. If degradation lasts >2 hours, escalate to the
model owner, not just infra on-call.

## 6. Postmortem (within 48h)
- Timeline, root cause, blast radius (rows served in heuristic mode),
  whether pages fired correctly, and one action item per contributing cause.
- Update this runbook if a new failure mode was found.

## Escalation
1. ML on-call (this runbook)
2. Model owner (fairness / long degradation)
3. Platform on-call (if matching is fully down, not just degraded)
