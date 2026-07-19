# Error Budget Report — PlaceMux Candidate-Job Ranking Service

Generated: 2026-07-19T15:27:53.995507+00:00

## SLO
- Availability target: **99.5%** over a **30-day** rolling window
- Allowed downtime budget: **216.0 minutes / 30 days**

## Evidence from this run (real logged requests, not hypothetical)
- Requests observed: 787
- Successes: 787 | Failures: 0 | Degraded/fallback: 482
- Actual availability this run: **100.0%**
- Observed traffic span: 68.6137 minutes
- Downtime-equivalent consumed this run: 42.0226 minutes
- Budget consumed (of the full 30-day budget): **19.4549%**
- Extrapolated burn rate: **122.49x** of the 30-day budget if this run's error rate held for the full window
- Budget exhausted: **False**

## Policy
- Burn rate > 2x for 1 hour -> page on-call (fast burn).
- Burn rate > 1x sustained for 6 hours -> ticket + freeze non-essential model deploys until budget recovers.
- Every alert in `logs/alerts.log` that maps to AVAILABILITY_BREACH or LATENCY_P99_HARD_BREACH counts against this budget.
- Degraded/fallback responses count as partial burn (service is 'up' but not meeting the quality bar users were promised) even though they don't 5xx.