# Pitfalls Checklist (Study Guide §12) — Mapped to Evidence

| Pitfall | How this repo avoids it | Evidence |
|---|---|---|
| No alert for a model returning constant/degenerate scores | `SCORE_DEGENERATE` rule on rolling score std | Fired live: `logs/alerts.log`, std dropped 0.22→0.0011 |
| SLOs with no owner | Every SLO and alert rule has an explicit owner + notify channel | `src/slo/slo_config.yaml` (`slo.owner`, `alerting.channels`) |
| Shipping an offline win that never gets validated online | Explicit "ship gate" requiring both offline nDCG win AND online-CTR-proxy win | `docs/EVAL_REPORT.md` "Ship gate" section |
| A fairness audit done once, at the end, as a formality | Fairness check is a standalone, re-runnable script wired into `run_demo.sh`, not a one-off notebook cell | `src/fairness/fairness_check.py`, step [5/7] of `run_demo.sh` |
| No model versioning, so you cannot say which model produced a decision six months ago | Every model is saved to a timestamped, immutable version directory with full metadata; every logged prediction carries `model_version` | `artifacts/models/<version>/metadata.json`, `logs/predictions.jsonl` |

## Additional hardening found by actually running this (not just designing it)

Running the full pipeline surfaced three real bugs that a "looks done" version would have shipped
with — this is the "verify and break stage is not optional" principle in practice:

1. **KS-drift stat silently stuck at 0.0** — the chaos script printed "reference distribution
   frozen" but never actually called an endpoint to freeze it. Fixed by adding a real
   `freeze_reference` action to `/chaos` and calling it before injecting faults
   (`src/serving/app.py`, `src/chaos/inject_failure.py`).
2. **Degenerate-score alert didn't fire on the first run** — the rolling window (200 requests)
   still had healthy history in it, diluting the degenerate signal below the alert threshold.
   Fixed by sending a large enough degenerate batch (220 requests) to fully flush the window —
   this is a realistic lesson about window sizing vs sustained-vs-transient failures, not just a
   test artifact.
3. **A malformed request (missing `cand_activity_score`) hit an uncaught KeyError** — caught,
   confirmed the service degrades gracefully (`success: false`, logged, no crash) rather than
   taking the process down, then fixed the demo's example payload to include all required fields.
