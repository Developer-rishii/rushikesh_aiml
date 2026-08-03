import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to sys path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import DATA_DIR
from serve import serve_ranking
from policy import PolicyStore
from preview import preview_config
from guardrails import validate_config

def test_pipeline():
    if not (DATA_DIR / "logs.pkl").exists() or not (DATA_DIR / "test_scored.pkl").exists():
        print("Data files missing, skipping tests.")
        return

    logs = pd.read_pickle(DATA_DIR / "logs.pkl")
    scored = pd.read_pickle(DATA_DIR / "test_scored.pkl")
    
    tenant = "acme_bank"
    job_id = scored[scored.tenant_id == tenant].job_id.iloc[0]
    store = PolicyStore()
    
    # 1. Test degraded mode
    result_down = serve_ranking(store, tenant, job_id, scored[scored.job_id == job_id], simulate_model_down=True)
    assert result_down["degraded_mode"] == True, "Expected degraded mode to be True"
    assert len(result_down["top10"]) > 0, "Expected non-empty results in degraded mode"
    print("PASS: Degraded mode works correctly.")
    
    # 2. Test guardrail rejection (fairness)
    bad_overrides = {"w_skill": 0.05, "w_experience": 0.05, "w_distance": 3.0, "min_skill_overlap": 0.95}
    preview_bad, proposed_bad = preview_config(store, tenant, bad_overrides, scored, sample_job_id=job_id)
    assert preview_bad["guardrail_passed"] == False, "Expected bad config to fail guardrails"
    print("PASS: Guardrails correctly reject unfair/nonsensical configs.")
    
    # 3. Test preview-before-commit
    good_overrides = {"w_skill": 0.8, "w_experience": 0.1, "w_distance": 0.1, "max_distance_km": 60.0}
    preview_good, proposed_good = preview_config(store, tenant, good_overrides, scored, sample_job_id=job_id)
    assert preview_good["guardrail_passed"] == True, "Expected good config to pass guardrails"
    
    # Verify commit changes live version
    initial_version = store.get(tenant).version
    store.commit(proposed_good, actor="test_admin")
    new_version = store.get(tenant).version
    assert new_version > initial_version, "Expected version to increment on commit"
    print("PASS: Preview-before-commit works correctly and increments version.")

if __name__ == "__main__":
    test_pipeline()
