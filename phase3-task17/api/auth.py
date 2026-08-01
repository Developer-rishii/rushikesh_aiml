"""
Partner authentication + quota/abuse protection.

Design decision (documented, not silent): strict fixed quotas PLUS a lightweight
anomaly signal (query diversity), rather than pure anomaly-detection. Rejected
pure-anomaly because it's non-deterministic to partners integrating against us -
they need a hard, documented number they can build retries around.
"""
import time
from collections import defaultdict, deque

# Partner registry: api_key -> (partner_name, plan)
PARTNERS = {
    "ats-demo-key-001": {"name": "DemoATS Inc.", "plan": "standard"},
    "ats-gold-key-002": {"name": "GoldPartner Corp.", "plan": "premium"},
}

PLAN_LIMITS = {
    "standard": {"requests_per_minute": 10, "requests_per_day": 2000},
    "premium": {"requests_per_minute": 60, "requests_per_day": 20000},
}


class QuotaStore:
    """In-memory token-bucket-ish store (sliding window). A real deployment
    swaps this for Redis; the interface is what matters for the API contract."""

    def __init__(self):
        self._minute_windows = defaultdict(deque)
        self._day_windows = defaultdict(deque)
        # abuse signal: track distinct candidate_ids queried per key in last 5 min
        self._recent_targets = defaultdict(deque)

    def check_and_record(self, api_key, target_id=None):
        plan = PARTNERS[api_key]["plan"]
        limits = PLAN_LIMITS[plan]
        now = time.time()

        mwin = self._minute_windows[api_key]
        while mwin and now - mwin[0] > 60:
            mwin.popleft()
        if len(mwin) >= limits["requests_per_minute"]:
            return False, "rate_limit_per_minute_exceeded", limits

        dwin = self._day_windows[api_key]
        while dwin and now - dwin[0] > 86400:
            dwin.popleft()
        if len(dwin) >= limits["requests_per_day"]:
            return False, "daily_quota_exceeded", limits

        mwin.append(now)
        dwin.append(now)

        if target_id is not None:
            tw = self._recent_targets[api_key]
            tw.append((now, target_id))
            while tw and now - tw[0][0] > 300:
                tw.popleft()
            distinct = len({t for _, t in tw})
            # extraction heuristic: querying a very large # of DISTINCT candidates
            # fast is the signature of someone scraping the model, not normal
            # ATS usage (which re-checks the same shortlist repeatedly).
            if distinct > 40 and len(tw) > 50:
                return False, "abuse_pattern_detected_broad_scraping", limits

        return True, None, limits

    def remaining(self, api_key):
        plan = PARTNERS[api_key]["plan"]
        limits = PLAN_LIMITS[plan]
        now = time.time()
        mwin = self._minute_windows[api_key]
        while mwin and now - mwin[0] > 60:
            mwin.popleft()
        return {
            "requests_remaining_this_minute": max(0, limits["requests_per_minute"] - len(mwin)),
            "requests_per_minute_limit": limits["requests_per_minute"],
        }


QUOTA = QuotaStore()


def authenticate(api_key):
    if not api_key or api_key not in PARTNERS:
        return None
    return PARTNERS[api_key]
