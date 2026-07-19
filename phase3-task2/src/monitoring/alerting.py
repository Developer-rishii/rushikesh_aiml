"""
Threshold-based alerting against the SLOs defined in src/slo/slo_config.yaml.
Evaluated on every rolling metrics snapshot. Each alert is written to
logs/alerts.log with severity + who gets paged, so "who gets paged and
what's the first thing they check" (a brainstorming question in the
study guide) has a concrete, demoable answer.
"""
import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "src" / "slo" / "slo_config.yaml"
ALERTS_LOG = ROOT / "logs" / "alerts.log"
ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class AlertEngine:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.fired_this_run = []

    def evaluate(self, snapshot: dict):
        """snapshot comes from MetricsStore.snapshot(). Returns list of
        alerts fired this evaluation (empty if system is healthy)."""
        slo = self.config["slo"]
        fired = []

        if snapshot["p95_latency_ms"] > slo["latency"]["target_ms"] and snapshot["n_requests_in_window"] >= 5:
            fired.append(self._make_alert(
                "LATENCY_P95_BREACH", "warning",
                f"p95 latency {snapshot['p95_latency_ms']}ms > SLO target {slo['latency']['target_ms']}ms",
                snapshot,
            ))

        if snapshot["p99_latency_ms"] > slo["latency"]["hard_ceiling_ms"] and snapshot["n_requests_in_window"] >= 5:
            fired.append(self._make_alert(
                "LATENCY_P99_HARD_BREACH", "page",
                f"p99 latency {snapshot['p99_latency_ms']}ms > hard ceiling {slo['latency']['hard_ceiling_ms']}ms",
                snapshot,
            ))

        if snapshot["score_std"] is not None and snapshot["score_std"] < slo["score_distribution"]["min_std_dev"]:
            fired.append(self._make_alert(
                "SCORE_DEGENERATE", "page",
                f"rolling score std {snapshot['score_std']} < min {slo['score_distribution']['min_std_dev']} "
                f"-- model may be returning near-constant scores (silent failure)",
                snapshot,
            ))

        if snapshot["ks_stat_vs_reference"] > slo["score_distribution"]["reference_ks_alert_threshold"]:
            fired.append(self._make_alert(
                "SCORE_DRIFT", "warning",
                f"KS stat vs reference {snapshot['ks_stat_vs_reference']} > "
                f"{slo['score_distribution']['reference_ks_alert_threshold']} -- score distribution has shifted",
                snapshot,
            ))

        if snapshot["success_rate_pct"] < slo["availability"]["target_pct"] and snapshot["n_requests_in_window"] >= 5:
            fired.append(self._make_alert(
                "AVAILABILITY_BREACH", "page",
                f"rolling success rate {snapshot['success_rate_pct']}% < SLO target "
                f"{slo['availability']['target_pct']}%",
                snapshot,
            ))

        for alert in fired:
            self._log_and_print(alert)
        self.fired_this_run.extend(fired)
        return fired

    def _make_alert(self, rule_id, severity, message, snapshot):
        owner = self.config["slo"]["owner"]
        channel = self.config["alerting"]["channels"]["page" if severity == "page" else "ticket"]
        return {
            "ts": time.time(),
            "rule_id": rule_id,
            "severity": severity,
            "message": message,
            "owner": owner,
            "notify_channel": channel,
            "snapshot": snapshot,
        }

    def _log_and_print(self, alert):
        with open(ALERTS_LOG, "a") as f:
            f.write(json.dumps(alert) + "\n")
        marker = "🚨 PAGE" if alert["severity"] == "page" else "⚠️  WARN"
        print(f"[{marker}] {alert['rule_id']}: {alert['message']}  -> {alert['notify_channel']}")
