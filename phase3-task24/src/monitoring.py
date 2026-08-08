"""
Minimal alerting layer. In production this would call PagerDuty/Opsgenie;
here it writes structured alert events so chaos tests can assert that a page
actually fired -- 'someone is paged, nothing silently breaks'.
"""
import json
import time


class Alerting:
    def __init__(self, log_path):
        self.log_path = log_path
        self.events = []

    def page(self, severity: str, reason: str, context: dict = None):
        event = {
            "ts": time.time(),
            "severity": severity,
            "reason": reason,
            "context": context or {},
        }
        self.events.append(event)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        return event
