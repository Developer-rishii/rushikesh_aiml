"""
Joiners / movers / leavers.

Design decision: role-change propagation is IMMEDIATE, not eventual.
Rejected alternative: eventual (batch, nightly) propagation. Rejected because
a recruiter who moves orgs mid-day and immediately starts using PlaceMux must
not see (or contaminate) their old org's context for even one session -- the
task's own bar is "leaves with them", and a nightly batch job violates that
bar for the entire day of the move. Immediate propagation costs a bit more
write amplification (recompute org aggregate on every move) but that is cheap
at PlaceMux's scale (tens of recruiters per org).
"""
from src.feature_store import ScopedFeatureStore


class IdentityLifecycleManager:
    def __init__(self, store: ScopedFeatureStore):
        self.store = store
        self.history = []  # audit trail: who moved where, when (ordinal step)

    def join(self, recruiter_id, org_id, step):
        self.store.set_membership(recruiter_id, org_id)
        self.history.append((step, recruiter_id, "join", None, org_id))

    def move(self, recruiter_id, new_org_id, step):
        old_org = self.store.current_org(recruiter_id)
        # Immediate propagation: membership flips atomically. Old org's
        # personal signal for this recruiter is purged (it must not keep
        # personalizing on the new org's behalf, and must not leak to the
        # old org's future recruiters via a stale key collision).
        if old_org is not None:
            self.store.purge_recruiter(recruiter_id, old_org)
        self.store.set_membership(recruiter_id, new_org_id)
        self.history.append((step, recruiter_id, "move", old_org, new_org_id))

    def leave(self, recruiter_id, step):
        org = self.store.current_org(recruiter_id)
        if org is not None:
            self.store.purge_recruiter(recruiter_id, org)
        self.history.append((step, recruiter_id, "leave", org, None))
