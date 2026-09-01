"""
run_serialize.py — Task 19's full flow, in the study guide's exact step
order:
  1. Bundle the fitted pipeline (preprocess + model) into one artifact.
  2. Save it with joblib/pickle plus a metadata file.
  3. Record library versions and training metrics.
  4. Write a load-and-predict function with input validation.
  5. Test loading in a fresh environment and predicting.
  6. Version the artifact for traceability.

Run: python -m src.run_serialize
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.train.build import train_and_evaluate
from src.serialize.metadata import build_metadata
from src.serialize.store import compute_version, save_artifact, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.run_serialize")


def main():
    t0 = time.time()
    cfg = load_config()
    log.info("Loaded config: %s", cfg)

    try:
        pipeline, metrics, features, split_sizes, X_test, y_test = train_and_evaluate(cfg)
    except (FileNotFoundError, ValueError) as e:
        log.error("Training stage failed: %s", e)
        sys.exit(1)

    artifact_version = compute_version(pipeline)
    metadata = build_metadata(cfg, metrics, features, split_sizes, artifact_version)

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    save_paths = save_artifact(pipeline, metadata, cfg)

    import subprocess
    fresh_env_script = Path(__file__).resolve().parent / "fresh_env_check.py"
    result_proc = subprocess.run(
        [sys.executable, str(fresh_env_script), str(cfg.store_dir), cfg.artifact_filename, cfg.metadata_filename,
         str(cfg.raw_data_path), str(cfg.locked_features_path), cfg.target_col],
        capture_output=True, text=True, timeout=60,
    )
    fresh_env_ok = result_proc.returncode == 0
    try:
        fresh_env_output = json.loads(result_proc.stdout.strip().splitlines()[-1])
    except Exception:
        fresh_env_output = {"raw_stdout": result_proc.stdout, "raw_stderr": result_proc.stderr}

    if not fresh_env_ok:
        log.error("[Step 5] Fresh-environment load-and-predict check FAILED: %s", result_proc.stderr)
    else:
        log.info("[Step 5] Fresh-environment load-and-predict check PASSED (subprocess, "
                  "no shared memory with training): %s", fresh_env_output)

    reordered_cols = list(reversed(features))
    reordered_result = predict(pipeline, metadata, X_test[reordered_cols])
    original_result = predict(pipeline, metadata, X_test)
    reorder_safe = reordered_result["predictions"] == original_result["predictions"]
    log.info("Column-reorder robustness check: predictions identical regardless of input column "
              "order = %s", reorder_safe)

    result = {
        "seed": cfg.seed,
        "training_metrics": metrics,
        "split_sizes": split_sizes,
        "artifact_version": artifact_version,
        "save_paths": save_paths,
        "metadata_summary": {
            "n_features": metadata["n_features"],
            "library_versions": metadata["library_versions"],
            "lineage": metadata["lineage"],
        },
        "fresh_environment_load_predict_check": {
            "passed": fresh_env_ok,
            "details": fresh_env_output,
        },
        "column_reorder_robustness_check": reorder_safe,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (cfg.report_dir / "serialize_report.json").write_text(json.dumps(result, indent=2, default=str))
    (cfg.log_dir / "run_serialize.log").write_text(json.dumps(result, indent=2, default=str))

    log.info("Done in %ss. Artifact version=%s. Fresh-env check=%s. Report -> %s",
              result["runtime_seconds"], artifact_version, fresh_env_ok,
              cfg.report_dir / "serialize_report.json")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
