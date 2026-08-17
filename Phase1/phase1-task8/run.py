#!/usr/bin/env python3
"""
run.py — Task 8's ONE COMMAND: the full end-to-end pipeline
(data -> features -> model -> evaluation -> output), per the study
guide's exact step order:
  1. Chain preprocessing + model into a single sklearn Pipeline.
  2. Wire data loading and splitting at the front.
  3. Attach evaluation and metric logging at the end.
  4. Make the whole thing run from one command.       <- this file
  5. Save model, preprocessor and metrics as run artifacts.
  6. Re-run to confirm identical results.              <- see --verify-reproducibility

Usage:
    python run.py                                # one run, default config
    python run.py --run-id my_run                # name the artifact folder
    python run.py --verify-reproducibility        # run twice, diff the metrics
"""
import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.loader import load_config
from src.data.dataset import load_dataframe, split_dataframe
from src.pipeline.build import build_pipeline
from src.pipeline.evaluate import evaluate_pipeline, log_run
from src.pipeline.artifacts import save_run_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("run")


def run_once(cfg, run_id: str) -> dict:
    """One full pass through every stage. Returns the result dict; also
    writes artifacts + experiment log rows as a side effect."""
    t0 = time.time()

    # ---- Step 2: data loading + splitting ----
    try:
        df = load_dataframe(cfg)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(df, cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error("[%s] Data stage failed: %s", run_id, e)
        sys.exit(1)
    split_sizes = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}

    # ---- Step 1: single sklearn Pipeline, fit ONCE (preprocessing + model together) ----
    try:
        pipeline = build_pipeline(cfg)
        pipeline.fit(X_train, y_train)
    except ValueError as e:
        log.error("[%s] Pipeline build failed: %s", run_id, e)
        sys.exit(1)
    except Exception as e:
        log.error("[%s] Pipeline fit raised an unexpected error: %s", run_id, e)
        sys.exit(1)
    log.info("[%s] [Step 1] Fitted single Pipeline (%s) on %s training rows.",
              run_id, [s[0] for s in pipeline.steps], len(X_train))

    # ---- Step 3: evaluation + metric logging ----
    try:
        metrics = evaluate_pipeline(pipeline, X_val, y_val, cfg.metrics)
    except Exception as e:
        log.error("[%s] Evaluation stage failed: %s", run_id, e)
        sys.exit(1)
    log.info("[%s] [Step 3] Validation metrics: %s", run_id, metrics)
    log_row = log_run(cfg.experiment_log_path, run_id, cfg, metrics, split_sizes)

    # ---- Step 5: save artifacts (one call — pipeline includes preprocessing) ----
    run_dir = cfg.artifact_dir / run_id
    run_metadata = {
        "run_id": run_id,
        "seed": cfg.seed,
        "model_name": cfg.model_name,
        "config_path": str(cfg.config_path),
        "split_sizes": split_sizes,
        "n_features": X_train.shape[1],
        "feature_names": list(X_train.columns),
    }
    artifact_paths = save_run_artifacts(run_dir, pipeline, metrics, run_metadata)
    log.info("[%s] [Step 5] Artifacts saved: %s", run_id, artifact_paths)

    result = {
        "run_id": run_id,
        "seed": cfg.seed,
        "split_sizes": split_sizes,
        "metrics": metrics,
        "artifact_paths": artifact_paths,
        "experiment_log_row": log_row,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    return result


def verify_reproducibility(cfg) -> dict:
    """Step 6: re-run to confirm identical results. Runs the ENTIRE
    pipeline twice, independently, from the same config and seed, and
    diffs the metrics byte-for-byte (not just eyeballed)."""
    log.info("=== Reproducibility check: running the full pipeline TWICE ===")
    result_a = run_once(cfg, run_id="run_a")
    result_b = run_once(cfg, run_id="run_b")

    identical = result_a["metrics"] == result_b["metrics"]
    diff = {k: (result_a["metrics"][k], result_b["metrics"][k])
            for k in result_a["metrics"] if result_a["metrics"][k] != result_b["metrics"][k]}

    verdict = {
        "identical_metrics": identical,
        "run_a_metrics": result_a["metrics"],
        "run_b_metrics": result_b["metrics"],
        "differences": diff,
    }
    if not identical:
        log.error("REPRODUCIBILITY CHECK FAILED: metrics differ between runs: %s", diff)
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        (cfg.log_dir / "reproducibility_check.json").write_text(json.dumps(verdict, indent=2))
        sys.exit(1)

    log.info("Reproducibility CONFIRMED: run_a and run_b produced byte-identical metrics: %s",
              result_a["metrics"])
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    (cfg.log_dir / "reproducibility_check.json").write_text(json.dumps(verdict, indent=2))
    return verdict


def main():
    parser = argparse.ArgumentParser(description="Task 8 — one-command end-to-end pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to a config YAML")
    parser.add_argument("--run-id", type=str, default="default_run", help="Name for this run's artifact folder")
    parser.add_argument("--verify-reproducibility", action="store_true",
                         help="Run the pipeline twice and confirm identical metrics (Step 6)")
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    log.info("Loaded config: %s", cfg)

    if args.verify_reproducibility:
        verdict = verify_reproducibility(cfg)
        print(json.dumps(verdict, indent=2))
    else:
        result = run_once(cfg, run_id=args.run_id)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
