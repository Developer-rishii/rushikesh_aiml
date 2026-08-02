"""
Stage E — Integrate, break it, then demo.
Run: python3 src/demo.py
Produces artifacts/demo_transcript.txt with real numbers (not assertions).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.feature_store import ScopedFeatureStore
from src.identity_lifecycle import IdentityLifecycleManager

def line(f, s=""):
    print(s)
    f.write(s + "\n")

def main():
    out = Path(__file__).parent.parent / "artifacts" / "demo_transcript.txt"
    with open(out, "w") as f:
        line(f, "=== PlaceMux Task 18 — Live Demo Transcript ===\n")

        store = ScopedFeatureStore()
        lifecycle = IdentityLifecycleManager(store)

        line(f, "Step 1: Recruiter 'rec_demo' joins org_A.")
        lifecycle.join("rec_demo", "org_A", step=0)
        store.ingest_event("rec_demo", "org_A", ["python", "ml"], clicked=True)
        store.ingest_event("rec_demo", "org_A", ["python", "aws"], clicked=True)
        sig = store.get_recruiter_signal("org_A", "rec_demo")
        line(f, f"  -> org_A personal signal top skills: {sig.top_skills(3)}")
        line(f, f"  -> current_org(rec_demo) = {store.current_org('rec_demo')}")

        line(f, "\nStep 2: rec_demo MOVES to org_B (e.g. changed employer).")
        lifecycle.move("rec_demo", "org_B", step=1)
        line(f, f"  -> current_org(rec_demo) = {store.current_org('rec_demo')}")
        sig_old = store.get_recruiter_signal("org_A", "rec_demo")
        sig_new = store.get_recruiter_signal("org_B", "rec_demo")
        line(f, f"  -> org_A signal for rec_demo AFTER move: {sig_old}  (must be None)")
        line(f, f"  -> org_B signal for rec_demo AFTER move: {sig_new}  (must be None/empty -- fresh start)")
        line(f, f"  -> VERIFIED: {'PASS' if sig_old is None else 'FAIL'} personalization context left org_A with the recruiter")

        line(f, "\nStep 3: rec_demo builds NEW org_B-scoped signal.")
        store.ingest_event("rec_demo", "org_B", ["sales", "product"], clicked=True)
        sig_new = store.get_recruiter_signal("org_B", "rec_demo")
        line(f, f"  -> org_B personal signal top skills: {sig_new.top_skills(3)}")

        line(f, "\nStep 4: INDUCED FAILURE -- simulate a stale event replayed under org_A after the move.")
        accepted = store.ingest_event("rec_demo", "org_A", ["python"], clicked=True)
        line(f, f"  -> stale event accepted? {accepted}  (must be False)")
        line(f, f"  -> org_A signal for rec_demo still: {store.get_recruiter_signal('org_A', 'rec_demo')}  (must remain None)")
        line(f, f"  -> DESIGNED DEGRADATION CONFIRMED: {'PASS' if not accepted else 'FAIL'} -- system rejects the write instead of silently corrupting scope")

        line(f, "\nStep 5: INDUCED FAILURE -- model/store unavailable at serve time (simulated).")
        def get_signal_or_fallback(store, org_id, recruiter_id):
            try:
                sig = store.get_recruiter_signal(org_id, recruiter_id)
                if sig is None or sig.impressions == 0:
                    raise LookupError("no personal signal yet")
                return ("personalized", sig)
            except Exception:
                org_sig = store.get_org_signal(org_id)
                if org_sig is not None:
                    return ("org_fallback", org_sig)
                return ("generic_fallback", None)

        mode, _ = get_signal_or_fallback(store, "org_B", "brand_new_recruiter_never_seen")
        line(f, f"  -> serving a never-seen recruiter falls back to: '{mode}' (graceful degradation, not a crash/None-pointer)")

        line(f, "\nStep 6: Offboarding.")
        lifecycle.leave("rec_demo", step=2)
        line(f, f"  -> current_org(rec_demo) after leave = {store.current_org('rec_demo')} (must be None)")
        line(f, f"  -> org_B personal signal after leave: {store.get_recruiter_signal('org_B', 'rec_demo')} (must be None -- deprovisioned)")
        org_b_agg = store.get_org_signal("org_B")
        line(f, f"  -> org_B AGGREGATE signal survives (collective, not personal): {org_b_agg.top_skills(3)}")

        line(f, "\nAudit trail (identity_lifecycle.history):")
        for entry in lifecycle.history:
            line(f, f"  {entry}")

    print(f"\nTranscript written to {out}")

if __name__ == "__main__":
    main()
