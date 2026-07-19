"""
Computes the error budget for the model service from the ACTUAL predictions
log (logs/predictions.jsonl) written by the live service during the demo run
-- not a hypothetical number. This is what turns "we have a 99.5%
availability SLO" into a documented, evidenced error budget report.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "src" / "slo" / "slo_config.yaml"
PREDICTIONS_LOG = ROOT / "logs" / "predictions.jsonl"
REPORT_PATH = ROOT / "docs" / "ERROR_BUDGET_REPORT.md"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def compute_error_budget():
    config = load_config()
    slo = config["slo"]
    target_pct = slo["availability"]["target_pct"]
    window_days = slo["window_days"]

    total_minutes_in_window = window_days * 24 * 60
    allowed_downtime_minutes = total_minutes_in_window * (1 - target_pct / 100)

    if not PREDICTIONS_LOG.exists():
        raise SystemExit("No predictions.jsonl found -- run the service and some traffic first.")

    records = [json.loads(l) for l in open(PREDICTIONS_LOG) if l.strip()]
    total = len(records)
    successes = sum(1 for r in records if r["success"])
    degraded = sum(1 for r in records if r.get("degraded"))
    failures = total - successes

    actual_availability_pct = 100.0 * successes / total if total else 100.0

    # Approximate "downtime minutes consumed" for this demo window by
    # attributing each failed/degraded request a slice of the window
    # proportional to its share of total traffic (a real system would use
    # wall-clock outage duration; here traffic is the unit of observation).
    span_seconds = (records[-1]["ts"] - records[0]["ts"]) if total > 1 else 0
    span_minutes = span_seconds / 60.0
    bad_fraction = (failures + degraded) / total if total else 0
    consumed_downtime_minutes = span_minutes * bad_fraction

    budget_consumed_pct = (
        100.0 * consumed_downtime_minutes / allowed_downtime_minutes
        if allowed_downtime_minutes > 0 else 0
    )

    # burn rate: how fast we're consuming the 30-day budget, extrapolated
    # from the observed window, expressed as "budgets per 30 days"
    if span_minutes > 0:
        burn_rate_per_30d = (consumed_downtime_minutes / span_minutes) * total_minutes_in_window / allowed_downtime_minutes
    else:
        burn_rate_per_30d = 0.0

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "slo_target_availability_pct": target_pct,
        "window_days": window_days,
        "allowed_downtime_minutes_per_window": round(allowed_downtime_minutes, 2),
        "observed_requests": total,
        "observed_successes": successes,
        "observed_failures": failures,
        "observed_degraded_fallback": degraded,
        "actual_availability_pct_this_run": round(actual_availability_pct, 3),
        "observed_span_minutes": round(span_minutes, 4),
        "consumed_downtime_minutes_this_run": round(consumed_downtime_minutes, 4),
        "budget_consumed_pct_of_30d_budget": round(budget_consumed_pct, 4),
        "extrapolated_burn_rate_x_of_30d_budget": round(burn_rate_per_30d, 2),
        "budget_exhausted": budget_consumed_pct >= 100,
    }
    return report


def write_report(report):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Error Budget Report — PlaceMux Candidate-Job Ranking Service",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## SLO",
        f"- Availability target: **{report['slo_target_availability_pct']}%** over a "
        f"**{report['window_days']}-day** rolling window",
        f"- Allowed downtime budget: **{report['allowed_downtime_minutes_per_window']} minutes / "
        f"{report['window_days']} days**",
        "",
        "## Evidence from this run (real logged requests, not hypothetical)",
        f"- Requests observed: {report['observed_requests']}",
        f"- Successes: {report['observed_successes']} | Failures: {report['observed_failures']} | "
        f"Degraded/fallback: {report['observed_degraded_fallback']}",
        f"- Actual availability this run: **{report['actual_availability_pct_this_run']}%**",
        f"- Observed traffic span: {report['observed_span_minutes']} minutes",
        f"- Downtime-equivalent consumed this run: {report['consumed_downtime_minutes_this_run']} minutes",
        f"- Budget consumed (of the full 30-day budget): **{report['budget_consumed_pct_of_30d_budget']}%**",
        f"- Extrapolated burn rate: **{report['extrapolated_burn_rate_x_of_30d_budget']}x** of the 30-day "
        f"budget if this run's error rate held for the full window",
        f"- Budget exhausted: **{report['budget_exhausted']}**",
        "",
        "## Policy",
        "- Burn rate > 2x for 1 hour -> page on-call (fast burn).",
        "- Burn rate > 1x sustained for 6 hours -> ticket + freeze non-essential model deploys "
        "until budget recovers.",
        "- Every alert in `logs/alerts.log` that maps to AVAILABILITY_BREACH or "
        "LATENCY_P99_HARD_BREACH counts against this budget.",
        "- Degraded/fallback responses count as partial burn (service is 'up' but not meeting "
        "the quality bar users were promised) even though they don't 5xx.",
    ]
    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    r = compute_error_budget()
    print(json.dumps(r, indent=2))
    write_report(r)
