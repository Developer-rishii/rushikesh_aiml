"""
Tests for Task 13. One test per named pitfall, plus a live end-to-end run
and edge cases.
Run: python tests/test_scoring_interface.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring.interface import Scorer, ScoringError
from src.scoring.registry import compute_model_version

ROOT = Path(__file__).resolve().parent.parent


def _real_record():
    import json
    import pandas as pd
    df = pd.read_csv(ROOT / "data" / "clean_from_task2.csv")
    features = json.loads((ROOT / "data" / "locked_feature_set.json").read_text())["final_feature_set"]
    for feat in features:
        if feat not in df.columns and feat.startswith("coeff_variation_"):
            m = feat.replace("coeff_variation_", "").replace("_", " ")
            mean_c, err_c = f"mean {m}", f"{m} error"
            if mean_c in df.columns and err_c in df.columns:
                df[feat] = df[err_c] / (df[mean_c].abs() + 1e-6)
    row = df[features].iloc[0]
    return {k.replace(" ", "_"): float(v) for k, v in row.items()}


def test_live_end_to_end_demo():
    from run_scoring_demo import main
    result = main()
    assert result["single_vs_batch_consistent"] is True
    assert Path(ROOT / "outputs" / "batch_scores" / "demo_batch_scores.csv").exists()
    print(f"PASS: live end-to-end demo — {result['n_records_scored']} real records scored, "
          f"{result['n_decisions_matching_ground_truth']}/{result['n_total']} decisions matched ground truth")


def test_pitfall_score_meaning_is_documented():
    """Pitfall: Undocumented score meaning."""
    scorer = Scorer()
    score = scorer.score_one(_real_record())
    assert len(score.score_meaning) > 30
    assert "benign" in score.score_meaning.lower() and "threshold" in score.score_meaning.lower()
    assert score.decision_label in ("benign", "malignant")
    print(f"PASS: every score carries an inline, human-readable meaning field, not a bare float — "
          f"e.g. decision_label='{score.decision_label}'")


def test_pitfall_input_validation_actually_rejects_bad_input():
    """Pitfall: No input validation."""
    scorer = Scorer()
    try:
        scorer.score_one({"mean_radius": 14.0})
        raised = False
    except ScoringError:
        raised = True
    assert raised, "an incomplete record was accepted — input validation is not actually enforced"

    good = _real_record()
    good["totally_made_up_field"] = 999
    try:
        scorer.score_one(good)
        raised2 = False
    except ScoringError:
        raised2 = True
    assert raised2, "an unexpected extra field was silently accepted — extra='forbid' is not working"

    bad_type = _real_record()
    bad_type["mean_radius"] = "not_a_number"
    try:
        scorer.score_one(bad_type)
        raised3 = False
    except ScoringError:
        raised3 = True
    assert raised3
    print("PASS: input validation actually rejects incomplete records, unexpected fields, AND wrong types "
          "(3 distinct malformed-input cases all caught)")


def test_pitfall_model_version_present_and_stable():
    """Pitfall: No model versioning on outputs."""
    scorer = Scorer()
    score = scorer.score_one(_real_record())
    assert score.model_version and score.model_version.startswith("sha256:")
    v1 = compute_model_version()
    v2 = compute_model_version()
    assert v1 == v2, "model version is not deterministic across calls"
    print(f"PASS: every score carries model_version={score.model_version}, "
          f"deterministically derived from the artifact's content hash")


def test_edge_case_missing_model_artifact_raises():
    from src.scoring import registry
    original = registry.MODEL_PATH
    try:
        registry.MODEL_PATH = Path("/tmp/does_not_exist_model.joblib")
        try:
            compute_model_version(registry.MODEL_PATH)
            raised = False
        except FileNotFoundError:
            raised = True
        assert raised
        print("PASS: missing model artifact raises a clear FileNotFoundError, not a silent failure")
    finally:
        registry.MODEL_PATH = original


def test_edge_case_empty_batch_raises():
    scorer = Scorer()
    try:
        scorer.score_batch([])
        raised = False
    except ScoringError:
        raised = True
    assert raised
    print("PASS: an empty batch request raises clearly instead of returning an empty/ambiguous result")


if __name__ == "__main__":
    test_pitfall_score_meaning_is_documented()
    test_pitfall_input_validation_actually_rejects_bad_input()
    test_pitfall_model_version_present_and_stable()
    test_edge_case_missing_model_artifact_raises()
    test_edge_case_empty_batch_raises()
    test_live_end_to_end_demo()
    print("\nALL TESTS PASSED")
