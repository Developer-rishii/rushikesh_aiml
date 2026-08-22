"""
run_ensemble.py — Task 11's full flow, in the study guide's exact step
order:
  1. Train a few diverse base models.
  2. Combine them via bagging/boosting/stacking.
  3. Evaluate the ensemble against the best single model.
  4. Check that diversity (not just duplication) is driving gains.
  5. Balance accuracy gain against added complexity/latency.
  6. Document the final ensemble and its lift.

Run: python -m src.run_ensemble
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.data.dataset import load_dataframe, split_dataframe
from src.models.base import build_all_base_pipelines
from src.models.ensemble import build_voting_ensemble, build_stacking_ensemble
from src.evaluation.metrics import evaluate_pipeline
from src.evaluation.diversity import compute_pairwise_disagreement, compute_error_overlap, diversity_verdict
from src.evaluation.latency import measure_inference_latency

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.run_ensemble")


def main():
    t0 = time.time()
    cfg = load_config()
    log.info("Loaded config: %s", cfg)

    # ---- data ----
    try:
        df = load_dataframe(cfg)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(df, cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error("Data stage failed: %s", e)
        sys.exit(1)

    # ---- Step 1: train diverse base models ----
    try:
        base_pipelines = build_all_base_pipelines(cfg)
        base_val_metrics, base_val_preds = {}, {}
        for name, pipe in base_pipelines.items():
            pipe.fit(X_train, y_train)
            base_val_metrics[name] = evaluate_pipeline(pipe, X_val, y_val, cfg.metrics)
            base_val_preds[name] = pipe.predict(X_val)
    except Exception as e:
        log.error("Base model stage failed: %s", e)
        sys.exit(1)
    log.info("[Step 1] Base model validation metrics: %s", base_val_metrics)

    best_single_name = max(base_val_metrics, key=lambda n: base_val_metrics[n][cfg.primary_metric])
    best_single_score = base_val_metrics[best_single_name][cfg.primary_metric]
    log.info("Best single model: %s (%s=%.4f)", best_single_name, cfg.primary_metric, best_single_score)

    # ---- Step 2: combine (voting + stacking), evaluate both ----
    try:
        voting = build_voting_ensemble(cfg)
        voting.fit(X_train, y_train)
        voting_val_metrics = evaluate_pipeline(voting, X_val, y_val, cfg.metrics)

        stacking = build_stacking_ensemble(cfg)
        stacking.fit(X_train, y_train)
        stacking_val_metrics = evaluate_pipeline(stacking, X_val, y_val, cfg.metrics)
    except ValueError as e:
        log.error("Ensemble build stage failed: %s", e)
        sys.exit(1)
    except Exception as e:
        log.error("Ensemble fit raised an unexpected error: %s", e)
        sys.exit(1)
    log.info("[Step 2] Voting metrics: %s | Stacking metrics: %s", voting_val_metrics, stacking_val_metrics)

    # ---- Step 3: evaluate ensembles against the best single model ----
    ensembles = {"voting": (voting, voting_val_metrics), "stacking": (stacking, stacking_val_metrics)}
    best_ensemble_name = max(ensembles, key=lambda n: ensembles[n][1][cfg.primary_metric])
    best_ensemble_pipeline, best_ensemble_metrics = ensembles[best_ensemble_name]
    lift = round(best_ensemble_metrics[cfg.primary_metric] - best_single_score, 4)
    log.info("[Step 3] Best ensemble=%s (%s=%.4f) vs best single=%s (%s=%.4f) -> lift=%+.4f",
              best_ensemble_name, cfg.primary_metric, best_ensemble_metrics[cfg.primary_metric],
              best_single_name, cfg.primary_metric, best_single_score, lift)

    # ---- Step 4: diversity check ----
    disagreement_df = compute_pairwise_disagreement(base_val_preds)
    overlap = compute_error_overlap(base_val_preds, y_val)
    diversity = diversity_verdict(overlap)

    # ---- Step 5: latency ----
    try:
        single_best_pipeline = base_pipelines[best_single_name]
        latency_single = measure_inference_latency(single_best_pipeline, X_val)
        latency_ensemble = measure_inference_latency(best_ensemble_pipeline, X_val)
    except ValueError as e:
        log.error("Latency measurement failed: %s", e)
        sys.exit(1)
    latency_overhead_pct = round(
        100 * (latency_ensemble["mean_batch_latency_ms"] - latency_single["mean_batch_latency_ms"])
        / max(latency_single["mean_batch_latency_ms"], 1e-9), 1
    )
    log.info("[Step 5] Latency overhead of ensemble vs best single model: %+.1f%%", latency_overhead_pct)

    # ---- Step 6: final decision + documented trade-offs ----
    keep_ensemble = lift >= cfg.min_lift_to_prefer_ensemble
    decision = (
        f"PREFER ensemble ({best_ensemble_name}): validated lift {lift:+.4f} >= "
        f"threshold {cfg.min_lift_to_prefer_ensemble}, latency overhead {latency_overhead_pct:+.1f}% accepted."
        if keep_ensemble else
        f"PREFER best single model ({best_single_name}): ensemble ({best_ensemble_name}) lift "
        f"{lift:+.4f} does not clear the {cfg.min_lift_to_prefer_ensemble} threshold — the added "
        f"latency ({latency_overhead_pct:+.1f}%) and complexity (multiple models to serve/maintain) "
        f"is not justified by the evidence."
    )
    log.info("[Step 6] %s", decision)

    kept_pipeline = best_ensemble_pipeline if keep_ensemble else single_best_pipeline
    kept_name = best_ensemble_name if keep_ensemble else best_single_name
    kept_test_metrics = evaluate_pipeline(kept_pipeline, X_test, y_test, cfg.metrics)

    # ---- persist ----
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    for name, pipe in base_pipelines.items():
        joblib.dump(pipe, cfg.artifact_dir / f"base_{name}.joblib")
    joblib.dump(voting, cfg.artifact_dir / "ensemble_voting.joblib")
    joblib.dump(stacking, cfg.artifact_dir / "ensemble_stacking.joblib")
    joblib.dump(kept_pipeline, cfg.artifact_dir / "kept_pipeline.joblib")
    disagreement_df.to_csv(cfg.report_dir / "pairwise_disagreement.csv", index=False)

    result = {
        "seed": cfg.seed,
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "base_model_val_metrics": base_val_metrics,
        "best_single_model": best_single_name,
        "best_single_score": best_single_score,
        "voting_val_metrics": voting_val_metrics,
        "stacking_val_metrics": stacking_val_metrics,
        "best_ensemble_strategy": best_ensemble_name,
        "primary_metric": cfg.primary_metric,
        "validated_lift": lift,
        "min_lift_to_prefer_ensemble": cfg.min_lift_to_prefer_ensemble,
        "diversity_check": {"error_overlap": overlap, "verdict": diversity},
        "latency": {"best_single_model": latency_single, "best_ensemble": latency_ensemble,
                    "overhead_pct": latency_overhead_pct},
        "decision": decision,
        "kept_model": kept_name,
        "kept_model_test_metrics": kept_test_metrics,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (cfg.report_dir / "ensemble_report.json").write_text(json.dumps(result, indent=2, default=str))
    (cfg.log_dir / "run_ensemble.log").write_text(json.dumps(result, indent=2, default=str))

    log.info("Done in %ss. Report -> %s", result["runtime_seconds"], cfg.report_dir / "ensemble_report.json")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
