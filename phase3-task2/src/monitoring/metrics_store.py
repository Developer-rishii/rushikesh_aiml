"""
Lightweight rolling-window metrics store for the model service.
Tracks per-request latency and prediction score in memory (bounded deque),
and computes the numbers the SLOs and alert rules are defined against:
p50/p95/p99 latency, rolling success rate (availability proxy), and
score-distribution stats used to catch a model returning confident
nonsense (silent failure) even though it "200 OK"s every request.
"""
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_LOG = LOG_DIR / "predictions.jsonl"


class MetricsStore:
    def __init__(self, window=200):
        self.window = window
        self.latencies_ms = deque(maxlen=window)
        self.scores = deque(maxlen=window)
        self.outcomes = deque(maxlen=window)  # True=success, False=error/fallback
        self.reference_scores = None  # frozen "known-good" distribution for drift KS test

    def freeze_reference(self):
        self.reference_scores = np.array(self.scores) if self.scores else None

    def record(self, latency_ms, score, success, model_version, degraded, extra=None):
        self.latencies_ms.append(latency_ms)
        if score is not None:
            self.scores.append(score)
        self.outcomes.append(success)
        rec = {
            "ts": time.time(),
            "latency_ms": latency_ms,
            "score": score,
            "success": success,
            "model_version": model_version,
            "degraded": degraded,
        }
        if extra:
            rec.update(extra)
        with open(PREDICTIONS_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def p(self, pct):
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(list(self.latencies_ms), pct))

    def p50(self):
        return self.p(50)

    def p95(self):
        return self.p(95)

    def p99(self):
        return self.p(99)

    def success_rate_pct(self):
        if not self.outcomes:
            return 100.0
        return 100.0 * sum(self.outcomes) / len(self.outcomes)

    def score_std(self):
        if len(self.scores) < 5:
            return None
        return float(np.std(list(self.scores)))

    def score_mean(self):
        if not self.scores:
            return None
        return float(np.mean(list(self.scores)))

    def ks_stat_vs_reference(self):
        """Two-sample KS statistic between current rolling scores and the
        frozen reference distribution -- flags distribution SHIFT even
        when nothing errored."""
        if self.reference_scores is None or len(self.scores) < 10:
            return 0.0
        from scipy import stats as _stats  # optional; fallback below if unavailable
        try:
            stat, _ = _stats.ks_2samp(self.reference_scores, np.array(self.scores))
            return float(stat)
        except Exception:
            return self._manual_ks()

    def _manual_ks(self):
        a = np.sort(self.reference_scores)
        b = np.sort(np.array(self.scores))
        all_vals = np.sort(np.concatenate([a, b]))
        cdf_a = np.searchsorted(a, all_vals, side="right") / len(a)
        cdf_b = np.searchsorted(b, all_vals, side="right") / len(b)
        return float(np.max(np.abs(cdf_a - cdf_b)))

    def snapshot(self):
        return {
            "n_requests_in_window": len(self.outcomes),
            "p50_latency_ms": round(self.p50(), 2),
            "p95_latency_ms": round(self.p95(), 2),
            "p99_latency_ms": round(self.p99(), 2),
            "success_rate_pct": round(self.success_rate_pct(), 3),
            "score_mean": round(self.score_mean(), 4) if self.score_mean() is not None else None,
            "score_std": round(self.score_std(), 4) if self.score_std() is not None else None,
            "ks_stat_vs_reference": round(self.ks_stat_vs_reference(), 4),
        }
