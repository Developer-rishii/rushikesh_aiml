# 2-Minute Live Demo Script (Stage E.4)

**0:00-0:20** — State the bar: "v2.0 must beat baseline on quality, pass
fairness, meet latency SLO, be cheap, be governed, and survive a live failure."

**0:20-0:50** — Show `reports/certification_pack.md`: nDCG/MAP lift table,
fairness gap PASS, p95 latency PASS, cost/1000, model card.

**0:50-1:20** — Show `reports/rollout_monitor_log.csv` +
`reports/rollback_decision.json`: walk the PSI(exp_years) curve, point at
day 20 where it crosses 0.25, show the rollback action it triggers.

**1:20-1:50** — Live failure injection: run
`python3 src/dr_failover.py` on stage, show it print fallback vs primary
quality — the system degrades to (never below) the old baseline, no request fails.

**1:50-2:00** — Close with the roadmap: 3 concrete Phase 4 items from
`reports/post_golive_report.md`.
