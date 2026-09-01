"""
Tests for Task 19. One test per named pitfall, plus a live end-to-end run
and edge cases.
Run: python tests/test_serialize.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.train.build import train_and_evaluate
from src.serialize.metadata import build_metadata
from src.serialize.store import compute_version, save_artifact, load_artifact, predict, ArtifactLoadError

_CACHE = {}


def _shared_train():
    if "pipeline" not in _CACHE:
        cfg = load_config()
        pipeline, metrics, features, split_sizes, X_test, y_test = train_and_evaluate(cfg)
        version = compute_version(pipeline)
        metadata = build_metadata(cfg, metrics, features, split_sizes, version)
        save_paths = save_artifact(pipeline, metadata, cfg)
        _CACHE.update(cfg=cfg, pipeline=pipeline, metadata=metadata, features=features,
                       X_test=X_test, y_test=y_test, save_paths=save_paths)
    return _CACHE["cfg"], _CACHE["pipeline"], _CACHE["metadata"], _CACHE["features"], _CACHE["X_test"], _CACHE["y_test"]


def test_live_end_to_end_run():
    from src.run_serialize import main
    result = main()
    assert result["fresh_environment_load_predict_check"]["passed"] is True
    assert result["column_reorder_robustness_check"] is True
    print(f"PASS: live end-to-end run — artifact_version={result['artifact_version']}, "
          f"fresh-env check passed, column-reorder-safe={result['column_reorder_robustness_check']}")


def test_pitfall_preprocessor_saved_with_model():
    """Pitfall: Saving model but not preprocessor."""
    cfg, pipeline, metadata, features, X_test, y_test = _shared_train()
    reloaded, reloaded_meta, mismatches = load_artifact(cfg.store_dir, cfg.artifact_filename, cfg.metadata_filename)
    step_names = [s[0] for s in reloaded.steps]
    assert "impute" in step_names and "scale" in step_names and "model" in step_names, (
        f"reloaded artifact is missing preprocessing steps — only found {step_names}"
    )
    raw_predictions = predict(reloaded, reloaded_meta, X_test)
    assert len(raw_predictions["predictions"]) == len(X_test)
    print(f"PASS: reloaded artifact contains preprocessing steps {step_names} bundled with the model — "
          f"raw unscaled input is scored correctly without any separate manual preprocessing")


def test_pitfall_version_mismatch_detected_not_silent():
    """Pitfall: Version mismatch breaking load."""
    cfg, pipeline, metadata, features, X_test, y_test = _shared_train()
    _, _, real_mismatches = load_artifact(cfg.store_dir, cfg.artifact_filename, cfg.metadata_filename)
    assert isinstance(real_mismatches, dict), "version mismatch check did not run"

    import json
    meta_path = cfg.store_dir / cfg.metadata_filename
    original = json.loads(meta_path.read_text())
    tampered = dict(original)
    tampered["library_versions"] = dict(original["library_versions"])
    tampered["library_versions"]["scikit-learn"] = "0.0.1-fake-old-version"
    meta_path.write_text(json.dumps(tampered))
    try:
        _, _, mismatches_after_tamper = load_artifact(cfg.store_dir, cfg.artifact_filename, cfg.metadata_filename)
        assert "scikit-learn" in mismatches_after_tamper, (
            "a deliberately tampered/mismatched scikit-learn version was not detected"
        )
        print(f"PASS: version-mismatch detector actually fires when versions genuinely differ: "
              f"{mismatches_after_tamper['scikit-learn']}")
    finally:
        meta_path.write_text(json.dumps(original))


def test_pitfall_metadata_lineage_present():
    """Pitfall: No metadata/lineage."""
    cfg, pipeline, metadata, features, X_test, y_test = _shared_train()
    required = {"artifact_version", "created_at_utc", "training_metrics", "library_versions",
                "lineage", "feature_names_ordered", "seed"}
    assert required <= metadata.keys(), f"metadata is missing required lineage fields: {required - metadata.keys()}"
    assert len(metadata["training_metrics"]) > 0
    print(f"PASS: metadata carries full lineage — version, timestamp, seed, training metrics "
          f"({metadata['training_metrics']}), library versions, and source data lineage")


def test_edge_case_missing_input_feature_raises():
    cfg, pipeline, metadata, features, X_test, y_test = _shared_train()
    incomplete = X_test.drop(columns=[features[0]])
    try:
        predict(pipeline, metadata, incomplete)
        raised = False
    except ArtifactLoadError:
        raised = True
    assert raised
    print("PASS: predicting with a missing required feature raises a clear ArtifactLoadError")


def test_edge_case_extra_unexpected_column_raises():
    cfg, pipeline, metadata, features, X_test, y_test = _shared_train()
    with_extra = X_test.copy()
    with_extra["totally_unexpected_column"] = 1.0
    try:
        predict(pipeline, metadata, with_extra)
        raised = False
    except ArtifactLoadError:
        raised = True
    assert raised
    print("PASS: predicting with an unexpected extra column raises clearly")


def test_edge_case_missing_artifact_file_raises():
    try:
        load_artifact(Path("/tmp/does_not_exist_artifact_store"), "model.joblib", "metadata.json")
        raised = False
    except ArtifactLoadError:
        raised = True
    assert raised
    print("PASS: loading from a missing store directory raises a clear ArtifactLoadError")


if __name__ == "__main__":
    test_pitfall_preprocessor_saved_with_model()
    test_pitfall_version_mismatch_detected_not_silent()
    test_pitfall_metadata_lineage_present()
    test_edge_case_missing_input_feature_raises()
    test_edge_case_extra_unexpected_column_raises()
    test_edge_case_missing_artifact_file_raises()
    test_live_end_to_end_run()
    print("\nALL TESTS PASSED")
