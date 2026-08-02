"""
Org- and recruiter-scoped personalization signals.

Design decision (see DESIGN_DECISIONS.md): signals are stored keyed by a
composite (org_id, recruiter_id) scope, NEVER by recruiter_id alone and NEVER
in one global table. This is what makes isolation testable: any lookup
requires both keys, so there is no code path that can accidentally return
another org's signal for the same recruiter_id string colliding across orgs,
and no code path that returns a recruiter's signal after they've changed org.

Two signal levels are kept, deliberately separate:
  - recruiter-level: this recruiter's personal click/shortlist affinity
  - org-level: aggregated affinity across all CURRENT recruiters of that org

Rejected alternative: a single flat dict keyed by recruiter_id, with org_id as
a field on the value. Rejected because it makes signal bleed a runtime bug
(easy to forget to filter by org) rather than a structural impossibility.
"""
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ScopedSignal:
    skill_affinity: dict = field(default_factory=lambda: defaultdict(float))
    impressions: int = 0
    clicks: int = 0

    def update(self, skills, clicked):
        self.impressions += 1
        if clicked:
            self.clicks += 1
            for s in skills:
                self.skill_affinity[s] += 1.0

    def top_skills(self, k=5):
        return sorted(self.skill_affinity.items(), key=lambda x: -x[1])[:k]


class ScopedFeatureStore:
    """Key = (org_id, recruiter_id). No global namespace exists."""

    def __init__(self):
        self._recruiter_signals: dict[tuple, ScopedSignal] = {}
        self._org_signals: dict[str, ScopedSignal] = {}
        # current membership -- the single source of truth for "who is in
        # which org right now". This is what identity_lifecycle.py mutates.
        self._current_org_of: dict[str, str] = {}

    # ---- membership -------------------------------------------------
    def set_membership(self, recruiter_id, org_id):
        self._current_org_of[recruiter_id] = org_id

    def current_org(self, recruiter_id):
        return self._current_org_of.get(recruiter_id)

    # ---- ingest -------------------------------------------------------
    def ingest_event(self, recruiter_id, org_id, skills, clicked):
        """Only ingest if org_id matches the recruiter's CURRENT membership.
        An event logged under a stale org (e.g. replayed after a move) must
        not update signals for an org the recruiter has left."""
        if self._current_org_of.get(recruiter_id) != org_id:
            return False  # rejected: stale/mismatched scope
        key = (org_id, recruiter_id)
        if key not in self._recruiter_signals:
            self._recruiter_signals[key] = ScopedSignal()
        self._recruiter_signals[key].update(skills, clicked)

        if org_id not in self._org_signals:
            self._org_signals[org_id] = ScopedSignal()
        self._org_signals[org_id].update(skills, clicked)
        return True

    # ---- read (always requires BOTH keys) ------------------------------
    def get_recruiter_signal(self, org_id, recruiter_id):
        """Returns None if recruiter is not currently in org_id -- this is
        the isolation guarantee: no cross-org read is possible."""
        if self._current_org_of.get(recruiter_id) != org_id:
            return None
        return self._recruiter_signals.get((org_id, recruiter_id))

    def get_org_signal(self, org_id):
        return self._org_signals.get(org_id)

    # ---- lifecycle: leave / offboard -----------------------------------
    def purge_recruiter(self, recruiter_id, org_id):
        """Deletion required on offboarding (DPDP + pitfall #1 in study guide:
        'Signals persisting after a user is deprovisioned'). Org-level
        aggregate signal is NOT deleted -- it's collective org behaviour, not
        personal data about the departed recruiter, and the study guide's own
        brainstorm question ('should org-level learning persist after
        everyone leaves?') is answered explicitly in DESIGN_DECISIONS.md."""
        self._recruiter_signals.pop((org_id, recruiter_id), None)
        if self._current_org_of.get(recruiter_id) == org_id:
            del self._current_org_of[recruiter_id]
