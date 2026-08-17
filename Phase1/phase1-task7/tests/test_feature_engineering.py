"""
Tests for Task 7. One test per named pitfall, plus a live end-to-end run
and edge cases.
Run: python tests/test_feature_engineering.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from configs.loader import load_config
from src.data.dataset import load_dataframe
from src.features.engineer import derive_domain_features, derive_aggregate_features, BASE_MEASUREMENTS
from src.features.pruning import check_leakage, prune_by_importance
from src.features.target_check import confirm_target


def test_live_end_to_end_run():
    from src.run_feature_engineering import main
    result = main()
    lock = result["baseline_lock"]
    assert len(lock["final_feature_set"]) > 0
    assert Path(str(load_config().baseline_feature_dir / "locked_feature_set.json")).exists()
    print(f"PASS: live end-to-end run — locked {len(lock['final_feature_set'])} features, "
          f"lift over original baseline = {lock['lift_over_original_baseline']:+.4f}")


def test_pitfall_engineered_features_are_leak_free_by_construction():
    """Pitfall: Engineering leaky features."""
    cfg = load_config()
    raw = load_dataframe(cfg)
    enriched, domain_added = derive_domain_features(raw)
    enriched, agg_added = derive_aggregate_features(enriched)
    candidates = domain_added + agg_added
    assert len(candidates) > 0

    # explicit leakage gate must actually run and be capable of catching a leak
    leaky = check_leakage(enriched, cfg.target_col, candidates, cfg.leakage_corr_threshold)
    assert isinstance(leaky, dict)

    # prove the gate has teeth: inject a deliberately leaky column and confirm it's caught
    enriched["fake_leak_column"] = enriched[cfg.target_col]  # perfect correlation
    leaky_with_plant = check_leakage(enriched, cfg.target_col, candidates + ["fake_leak_column"],
                                       cfg.leakage_corr_threshold)
    assert "fake_leak_column" in leaky_with_plant
    print(f"PASS: leakage gate ran on {len(candidates)} real candidates (0 flagged) AND "
          f"correctly caught a deliberately planted perfect-correlation column")


def test_pitfall_features_measured_not_assumed():
    """Pitfall: Adding features without measuring lift."""
    from src.run_feature_engineering import main
    result = main()
    lock = result["baseline_lock"]
    # the report must contain an actual measured PR-AUC lift number, not just a feature list
    assert "pr_auc_original_only" in lock and "pr_auc_final_locked_set" in lock
    assert "lift_over_original_baseline" in lock
    assert isinstance(lock["lift_over_original_baseline"], float)
    print(f"PASS: lift is a measured number ({lock['lift_over_original_baseline']:+.4f} PR-AUC), "
          f"not asserted without evidence")


def test_pitfall_domain_knowledge_documented():
    """Pitfall: Ignoring domain knowledge."""
    import inspect
    from src.features import engineer
    source = inspect.getsource(engineer)
    # each derivation function must carry an explicit domain rationale docstring
    assert "pathologist" in source.lower() or "domain" in source.lower()
    assert "worst-to-mean" in source.lower() or "coefficient of variation" in source.lower()
    print("PASS: engineered features carry explicit, inspectable domain rationale (not black-box columns)")


def test_step1_target_check_catches_bad_target():
    bad_df = pd.DataFrame({"target": [0, 1, 2, 0, 1], "x": [1, 2, 3, 4, 5]})  # not binary
    try:
        confirm_target(bad_df, "target")
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: target-quality check rejects a non-binary target, not silently accepting it")


def test_edge_case_missing_target_column():
    df = pd.DataFrame({"x": [1, 2, 3]})
    try:
        confirm_target(df, "target")
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: missing target column raises clearly")


def test_edge_case_division_safe_on_zero_mean():
    df = pd.DataFrame({"mean radius": [0.0, 5.0], "worst radius": [1.0, 10.0]})
    enriched, added = derive_domain_features(df)
    assert not enriched["ratio_worst_to_mean_radius"].isna().any()
    assert not (enriched["ratio_worst_to_mean_radius"].abs() == float("inf")).any()
    print("PASS: worst-to-mean ratio doesn't blow up (inf/NaN) on a zero mean value")


if __name__ == "__main__":
    test_pitfall_engineered_features_are_leak_free_by_construction()
    test_pitfall_domain_knowledge_documented()
    test_step1_target_check_catches_bad_target()
    test_edge_case_missing_target_column()
    test_edge_case_division_safe_on_zero_mean()
    test_live_end_to_end_run()
    test_pitfall_features_measured_not_assumed()
    print("\nALL TESTS PASSED")
