"""
Stage B.4 / Stage E.3 - "what happens when the model is unavailable" +
"deliberately induce the failure and confirm the designed degradation".

We simulate candidate_v2 becoming unavailable (e.g. inference service
timeout) on a specific day, and prove the serving layer falls back to
baseline_v1 rather than erroring out or serving garbage.
"""


class ModelUnavailable(Exception):
    pass


class FlakyModel:
    """Wraps a TrainedModel; raises on the configured failure day(s)."""

    def __init__(self, model, fail_on_days):
        self.model = model
        self.fail_on_days = set(fail_on_days)

    def score(self, df, day: int):
        if day in self.fail_on_days:
            raise ModelUnavailable(
                f"{self.model.name} {self.model.version} inference service "
                f"unavailable on day {day}"
            )
        return self.model.score(df)


def score_with_fallback(flaky_model, fallback_model, df, day: int, event_log: list):
    """Try the primary (possibly flaky) model; fall back safely on failure."""
    try:
        return flaky_model.score(df, day), "candidate_v2"
    except Exception as e:  # noqa: BLE001 - deliberately broad: any failure -> fallback
        event_log.append({"day": day, "event": "fallback_triggered", "detail": str(e)})
        return fallback_model.score(df), "baseline_v1 (fallback)"
