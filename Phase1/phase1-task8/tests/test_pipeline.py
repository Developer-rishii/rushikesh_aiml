"""
Tests for Task 8. One test per named pitfall, plus a live end-to-end run
(including the reproducibility check) and edge cases.
Run: python tests/test_pipeline.py
"""
import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inspect
import joblib

from configs.loader import load_config
from src.data.dataset import load_dataframe, split_dataframe, load_locked_features
from src.pipeline.build import build_pipeline, MODEL_REGISTRY
from src.pipeline.artifacts import save_run_artifacts, load_run_pipeline

ROOT = Path(__file__).resolve().parent.parent


def test_pitfall_preprocessing_is_inside_the_pipeline_object():
    """Pitfall: Preprocessing applied outside the pipeline."""
    from sklearn.pipeline import Pipeline
    cfg = load_config()
    pipeline = build_pipeline(cfg)
    assert isinstance(pipeline, Pipeline)
    step_names = [s[0] for s in pipeline.steps]
    assert "impute" in step_names and "model" in step_names
    # structural proof, not just naming: confirm run.py never calls a
    # preprocessing step's fit_transform separately from pipeline.fit
    source = inspect.getsource(__import__("run"))
    assert "fit_transform" not in source, (
        "run.py calls fit_transform directly — preprocessing would be "
        "escaping the single Pipeline object"
    )
    assert "pipeline.fit(X_train" in source
    print("PASS: preprocessing lives INSIDE the single Pipeline object; "
          "run.py never fits it separately")


def test_pitfall_runs_are_reproducible():
    """Pitfall: Non-reproducible runs."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--verify-reproducibility"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"reproducibility check failed:\n{result.stdout}\n{result.stderr}"
    import json
    verdict = json.loads(result.stdout)
    assert verdict["identical_metrics"] is True
    assert verdict["differences"] == {}
    print(f"PASS: two independent full runs produced byte-identical metrics: {verdict['run_a_metrics']}")


def test_pitfall_artifacts_are_actually_saved():
    """Pitfall: No saved artifacts."""
    cfg = load_config()
    df = load_dataframe(cfg)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(df, cfg)
    pipeline = build_pipeline(cfg)
    pipeline.fit(X_train, y_train)

    from src.pipeline.evaluate import evaluate_pipeline
    metrics = evaluate_pipeline(pipeline, X_val, y_val, cfg.metrics)

    test_run_dir = cfg.artifact_dir / "test_artifact_check"
    paths = save_run_artifacts(test_run_dir, pipeline, metrics, {"run_id": "test_artifact_check"})

    assert Path(paths["pipeline_path"]).exists()
    assert Path(paths["metrics_path"]).exists()
    assert Path(paths["metadata_path"]).exists()

    reloaded = load_run_pipeline(test_run_dir)
    reloaded_metrics = evaluate_pipeline(reloaded, X_val, y_val, cfg.metrics)
    assert reloaded_metrics == metrics, "reloaded pipeline gives different metrics than the original"
    print("PASS: pipeline.joblib + metrics.json + run_metadata.json all saved and reload to identical results")

    import shutil
    shutil.rmtree(test_run_dir)


def test_live_end_to_end_one_command_run():
    result = subprocess.run(
        [sys.executable, str(ROOT / "run.py"), "--run-id", "live_test_run"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"one-command run failed:\n{result.stdout}\n{result.stderr}"
    import json
    output = json.loads(result.stdout)
    assert "metrics" in output and output["metrics"]["pr_auc"] > 0
    assert Path(output["artifact_paths"]["pipeline_path"]).exists()
    print(f"PASS: single `python run.py` command produced a model + metrics: {output['metrics']}")


def test_edge_case_schema_drift_locked_feature_missing():
    """Simulates what breaks if the input schema changes slightly (a
    brainstorming question the guide explicitly asks)."""
    import pandas as pd
    cfg = load_config()
    df = pd.read_csv(cfg.raw_data_path)
    df = df.drop(columns=["mean radius"])  # simulate a locked feature vanishing
    tmp_path = ROOT / "data" / "_tmp_schema_drift.csv"
    df.to_csv(tmp_path, index=False)
    try:
        cfg.raw_data_path = tmp_path
        try:
            load_dataframe(cfg)
            raised = False
        except ValueError:
            raised = True
        assert raised, "schema drift (missing locked feature) should raise clearly, not silently proceed"
        print("PASS: schema drift (a locked feature going missing) raises clearly, doesn't fail silently")
    finally:
        tmp_path.unlink(missing_ok=True)


def test_edge_case_unknown_model_raises():
    cfg = load_config()
    cfg.model_name = "does_not_exist"
    try:
        build_pipeline(cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: unknown model name raises a clear error before any fitting is attempted")


if __name__ == "__main__":
    test_pitfall_preprocessing_is_inside_the_pipeline_object()
    test_pitfall_artifacts_are_actually_saved()
    test_edge_case_schema_drift_locked_feature_missing()
    test_edge_case_unknown_model_raises()
    test_live_end_to_end_one_command_run()
    test_pitfall_runs_are_reproducible()
    print("\nALL TESTS PASSED")
