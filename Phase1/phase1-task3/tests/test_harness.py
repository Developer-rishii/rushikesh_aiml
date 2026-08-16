"""
Tests for the Task 3 skeleton: live end-to-end run, edge cases, and proof
that swapping models is config-only (no code touched).
Run: python tests/test_harness.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.harness import run
from src.models.registry import build_model


def test_baseline_run_end_to_end():
    result = run(Path(__file__).resolve().parent.parent / "configs" / "config.yaml")
    assert result["metrics"]["pr_auc"] > 0
    assert Path(result["model_path"]).exists()
    print("PASS: baseline (logreg) end-to-end run, artifact saved")


def test_model_swap_via_config_only():
    rf_config = Path(__file__).resolve().parent.parent / "configs" / "config_random_forest.yaml"
    result = run(rf_config)
    assert result["model_name"] == "random_forest"
    assert result["metrics"]["pr_auc"] > 0
    print("PASS: random_forest run via a config swap only — no source file touched")


def test_experiment_log_accumulates_both_runs():
    cfg = load_config()
    log_path = cfg.experiment_log_path
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    # header + at least the 2 runs from the tests above
    assert len(lines) >= 3
    assert "logreg_baseline" in log_path.read_text()
    assert "random_forest" in log_path.read_text()
    print("PASS: experiment log accumulated rows for both runs, not overwritten")


def test_unknown_model_raises_clear_error():
    try:
        build_model("does_not_exist", {})
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: unknown model name in registry raises a clear error")


def test_missing_config_raises():
    try:
        load_config(Path("/tmp/nonexistent_config_12345.yaml"))
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
    print("PASS: missing config file raises clearly, not a silent failure")


def test_bad_split_fractions_raise():
    import yaml, tempfile, os
    cfg_dict = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs" / "config.yaml").read_text()
    )
    cfg_dict["data"]["train_frac"] = 0.5  # now sums to 0.85, invalid
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg_dict, tmp)
        tmp_path = tmp.name
    try:
        from src.data.dataset import load_dataframe, split_dataframe
        cfg = load_config(Path(tmp_path))
        df = load_dataframe(cfg)
        try:
            split_dataframe(df, cfg)
            raised = False
        except ValueError:
            raised = True
        assert raised
        print("PASS: invalid split fractions (don't sum to 1.0) raise clearly")
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    test_baseline_run_end_to_end()
    test_model_swap_via_config_only()
    test_experiment_log_accumulates_both_runs()
    test_unknown_model_raises_clear_error()
    test_missing_config_raises()
    test_bad_split_fractions_raise()
    print("\nALL TESTS PASSED")
