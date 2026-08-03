"""
Stage E — Integrate, break it, then demo.
Run: python3 demo.py   (from src/)
This is the exact script to run live for the 2-minute demo. It:
  1. Runs the full pipeline end-to-end on real logged data for a tenant.
  2. Changes that tenant's config LIVE via preview -> guardrail -> commit,
     and shows the ranked results change.
  3. Attempts a config that IS unfair/nonsensical and shows guardrail
     rejection (nothing goes live).
  4. Deliberately induces the "model unavailable" failure and confirms
     the designed degradation (fallback score, degraded_mode flag, no crash).
"""
import pandas as pd
from policy import PolicyStore
from guardrails import validate_config
from preview import preview_config
from serve import serve_ranking

pd.set_option("display.width", 120)


def main():
    from config import DATA_DIR
    if not (DATA_DIR / "logs.pkl").exists() or not (DATA_DIR / "test_scored.pkl").exists():
        print("ERROR: Missing data files. Please run data_gen.py and model.py first.")
        import sys; sys.exit(1)
        
    logs = pd.read_pickle(DATA_DIR / "logs.pkl")
    scored = pd.read_pickle(DATA_DIR / "test_scored.pkl")
    store = PolicyStore()

    tenant = "acme_bank"
    job_id = scored[scored.tenant_id == tenant].job_id.iloc[0]

    print("=" * 70)
    print("STEP 1 — baseline live ranking (default policy config)")
    print("=" * 70)
    result_before = serve_ranking(store, tenant, job_id, scored[scored.job_id == job_id])
    print(result_before["top10"].to_string(index=False))
    print(result_before["explanation"])

    print()
    print("=" * 70)
    print("STEP 2 — admin previews a NEW config for acme_bank (raise skill weight,"
          " tighten distance) BEFORE committing")
    print("=" * 70)
    good_overrides = {"w_skill": 0.8, "w_experience": 0.1, "w_distance": 0.1,
                       "max_distance_km": 60.0}
    preview, proposed_good = preview_config(store, tenant, good_overrides, scored,
                                             sample_job_id=job_id)
    print(f"guardrail_passed={preview['guardrail_passed']}  "
          f"violations={preview['guardrail_violations']}")
    print("fairness selection rates by group (audit-only attribute):",
          preview["fairness_selection_rates"])
    print("funnel impact:", preview["funnel_impact"])
    print("--- BEFORE (top 5) ---")
    print(preview["before_top10"].head(5).to_string(index=False))
    print("--- AFTER preview (top 5) ---")
    print(preview["after_top10"].head(5).to_string(index=False))

    assert preview["guardrail_passed"], "expected this config to pass guardrails"
    store.commit(proposed_good, actor="demo_admin")
    print(f"\n>>> COMMITTED config version {proposed_good.version} for {tenant} <<<")

    print()
    print("=" * 70)
    print("STEP 3 — live re-rank AFTER committing the new config (real change, not"
          " just a preview)")
    print("=" * 70)
    result_after = serve_ranking(store, tenant, job_id, scored[scored.job_id == job_id])
    print(result_after["top10"].to_string(index=False))
    print(result_after["explanation"])
    changed = not result_before["top10"].reset_index(drop=True).equals(
        result_after["top10"].reset_index(drop=True))
    print(f"\nRanking actually changed after live config update: {changed}")

    print()
    print("=" * 70)
    print("STEP 4 — admin tries a BAD config (nonsensical weights + would encode"
          " historical bias) -> guardrail must reject, nothing goes live")
    print("=" * 70)
    bad_overrides = {"w_skill": 0.05, "w_experience": 0.05, "w_distance": 3.0,
                      "min_skill_overlap": 0.95}
    preview_bad, proposed_bad = preview_config(store, tenant, bad_overrides, scored,
                                                sample_job_id=job_id)
    print(f"guardrail_passed={preview_bad['guardrail_passed']}")
    for v in preview_bad["guardrail_violations"]:
        print("  REJECTED:", v)
    live_version_before = store.get(tenant).version
    if not preview_bad["guardrail_passed"]:
        print(">>> Commit BLOCKED. Live config unchanged. <<<")
    else:
        store.commit(proposed_bad, actor="demo_admin")
    live_version_after = store.get(tenant).version
    print(f"live config version unchanged: {live_version_before == live_version_after}")

    print()
    print("=" * 70)
    print("STEP 5 — deliberately induce model-unavailable failure")
    print("=" * 70)
    result_down = serve_ranking(store, tenant, job_id, scored[scored.job_id == job_id],
                                 simulate_model_down=True)
    print(f"degraded_mode={result_down['degraded_mode']}  "
          f"(pipeline kept serving instead of crashing/500ing)")
    print(result_down["top10"].head(5).to_string(index=False))

    print()
    print("=" * 70)
    print("STEP 6 — try to commit a config for a DIFFERENT tenant fairness-audited"
          " on real biased-log data (four-fifths rule)")
    print("=" * 70)
    unfair_overrides = {"w_skill": 1.0, "w_experience": 0.0, "w_distance": 0.0}
    # This alone isn't inherently unfair; the real check is the audited
    # selection-rate ratio on the actual (historically biased) logs.
    preview_c, proposed_c = preview_config(store, "orion_retail", unfair_overrides, scored)
    print("orion_retail proposal guardrail_passed:", preview_c["guardrail_passed"])
    print("selection rates:", preview_c["fairness_selection_rates"])
    if preview_c["guardrail_violations"]:
        for v in preview_c["guardrail_violations"]:
            print("  REJECTED:", v)


if __name__ == "__main__":
    main()
