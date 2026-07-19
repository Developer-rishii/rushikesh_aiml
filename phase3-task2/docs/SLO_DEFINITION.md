# Inference SLOs — PlaceMux Candidate-Job Ranking Service

Owner: `ml-oncall@altrodav.example` · Service: candidate↔job ranking model · Window: 30 days

## 1. What good looks like (the bar)

A model that is **slow or silently returning garbage should page someone before users notice.**
Concretely, three things must all be true, continuously, not just at deploy time:

| SLO | Target | Why this number |
|---|---|---|
| **Inference latency (p95)** | ≤ 150 ms | This sits behind an interactive search/recommendation page; p95 (not p50) is what a meaningful fraction of real users actually feel, since ranking calls fan out across many impressions per page load. |
| **Inference latency (p99, hard ceiling)** | ≤ 400 ms | Above this, page load timeouts start truncating result lists. p99 breach pages immediately — it's a harder, rarer, more serious threshold than p95. |
| **Availability** | ≥ 99.5% / 30 days | 216 minutes/month allowed downtime. Matches the platform-wide SLO DevOps already runs other services against (see hand-off, §13 of the study guide). |
| **Prediction-quality floor (offline)** | nDCG@10 ≥ 0.60 | Below this, ranking is not meaningfully better than showing jobs in an arbitrary order; a retrain/rollback should trigger. |
| **Prediction-quality floor (online proxy)** | ≥ +5% CTR uplift vs baseline | Offline wins that don't show up online are not shipped (see Ship Gate in `docs/EVAL_REPORT.md`). |

## 2. The baseline every claim is measured against

**Popularity baseline**: rank jobs purely by `job_popularity` (a heavy-tailed, "how often is this job
shown/applied to" signal already in the logs). This is the simplest thing that could possibly work.
The model must beat it on held-out nDCG@10 — not just perform "okay" in isolation. See
`docs/EVAL_REPORT.md` for the actual numbers from this run (model 0.8008 vs baseline 0.7342).

## 3. Why p95 (not p99) is the *target* metric, and p99 is the *hard ceiling*

- p95 is what most real users feel on an interactive page — a good day-to-day operating target.
- p99 catches the tail that p95 would hide: a small fraction of very slow requests that, if ignored,
  quietly degrade the experience for the unluckiest 1% of traffic every single day. It is set as a
  **hard ceiling** that pages immediately rather than a soft target, because by the time p99 breaches,
  users are already seeing timeouts.

## 4. What "silently returning garbage" means operationally

Two independent monitors catch this, because a broken model that still returns HTTP 200 with a
plausible-looking float is worse than one that errors:

1. **Score-distribution collapse** — rolling standard deviation of predicted scores falls below
   `0.03`. A model returning near-constant scores has stopped discriminating between good and bad
   matches even though every request "succeeds".
2. **Distribution drift (KS test)** — the rolling score distribution is compared against a frozen
   reference distribution (taken right after the last validated deploy). A KS statistic above `0.25`
   means the shape of the output has shifted — something upstream broke (a feature pipeline change,
   a schema drift in `job_popularity`, a stale feature store) even if nothing threw an exception.

Both are demonstrated firing against real, live traffic in `logs/alerts.log` — see
`docs/WORKED_EXAMPLE.md`.

## 5. Who gets paged, and what they check first

1. **Page fires** (`SCORE_DEGENERATE`, `LATENCY_P99_HARD_BREACH`, `AVAILABILITY_BREACH`) →
   `ml-oncall@altrodav.example` via PagerDuty (`placemux-ml-service`).
2. **First checks, in order**:
   - `GET /health` — is the correct model version loaded? Does `feature_pipeline_version` match what
     was last deployed?
   - `GET /metrics` — current rolling p50/p95/p99, score mean/std, KS-vs-reference.
   - `logs/predictions.jsonl` (tail) — are individual scores plausible for their inputs, or all
     converging to one value?
   - Is the service in `degraded=true` (fallback) mode already? If so, the popularity fallback is
     absorbing user-facing impact while the primary model is fixed — check `docs/ERROR_BUDGET_REPORT.md`
     for how much budget that's burning.
3. **Warnings** (`LATENCY_P95_BREACH`, `SCORE_DRIFT`) go to `#ml-platform-alerts` (Slack) for the
   next business-hours look, not an overnight page.

## 6. Alternative approaches considered (and rejected)

- **Latency SLO on the model service alone vs end-to-end user-perceived latency** — we chose the
  model-service-only latency (measured at the `/predict` boundary) because it is what this team
  owns and can act on directly; end-to-end page latency also includes UI rendering and network time
  that live outside this system's control and would make the SLO un-actionable for this team. The
  end-to-end number is still tracked by the frontend team as their own SLO.
- **Full drift detection (e.g., population stability index across every feature) vs score-distribution
  monitoring now** — we chose score-distribution monitoring first because it is the cheapest signal
  that catches the broadest class of "silent failure" (anything that breaks upstream shows up as a
  score shift), and layered per-feature drift detection is called out as future work in
  `docs/MODEL_CARD.md` rather than gold-plating this iteration.
