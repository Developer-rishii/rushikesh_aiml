"""
run_experiment.py — Task 5's full flow, in the brief's exact step order:
  1. Compute a baseline metric.
  2. Train a simple first model through the harness.
  3. Evaluate on validation with the chosen metric.
  4. Compare against the baseline — is it actually better?
  5. Inspect the worst errors for patterns.
  6. Record the run and decide the next improvement.

Both the baseline and the real model reuse the EXACT SAME preprocessing
pipeline (fit once, on train only — Task 4's deliverable) and the exact
same metrics module, so the comparison in step 4 is apples-to-apples.

Run: python -m src.run_experiment
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.data.dataset import load_dataframe, enrich_dataframe, split_dataframe
from src.preprocessing.pipeline import fit_preprocessor, transform, verify_no_leakage, save_preprocessor
from src.modeling.baseline import build_baseline
from src.modeling.registry import build_model
from src.modeling.metrics import compute_metrics
from src.modeling.errors import worst_errors, summarize_error_patterns
from src.modeling.experiment_log import log_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.run_experiment")


def main():
    t0 = time.time()
    cfg = load_config()
    log.info("Loaded config: %s", cfg)

    # ---- data ----
    try:
        raw = load_dataframe(cfg)
        enriched = enrich_dataframe(raw, cfg)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(enriched, cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error("Data stage failed: %s", e)
        sys.exit(1)
    split_sizes = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}

    # ---- preprocessing (fit ONCE on train, reused for baseline + model) ----
    try:
        preprocessor = fit_preprocessor(X_train, cfg)
        verify_no_leakage(preprocessor, X_train, X_val)
        X_train_t = transform(preprocessor, X_train)
        X_val_t = transform(preprocessor, X_val)
    except (ValueError, RuntimeError) as e:
        log.error("Preprocessing stage failed: %s", e)
        sys.exit(1)
    save_preprocessor(preprocessor, cfg.model_dir / "fitted_preprocessor.joblib")

    # ---- Step 1: baseline metric ----
    try:
        baseline_clf = build_baseline(cfg)
        baseline_clf.fit(X_train_t, y_train)
        baseline_proba = baseline_clf.predict_proba(X_val_t)[:, 1]
        baseline_pred = baseline_clf.predict(X_val_t)
        baseline_metrics = compute_metrics(y_val, baseline_proba, baseline_pred, cfg.metrics)
    except Exception as e:
        log.error("Baseline stage failed: %s", e)
        sys.exit(1)
    log.info("[Step 1] Baseline (%s) validation metrics: %s", cfg.baseline_strategy, baseline_metrics)
    baseline_row = log_run(cfg.experiment_log_path, "baseline", cfg, baseline_metrics, split_sizes,
                            extra={"model_name": "dummy_baseline"})

    # ---- Step 2: train the first real model, through the same harness ----
    try:
        model = build_model(cfg.model_name, cfg.model_params)
        model.fit(X_train_t, y_train)
    except ValueError as e:
        log.error("Model stage failed: %s", e)
        sys.exit(1)
    except Exception as e:
        log.error("Model training raised an unexpected error: %s", e)
        sys.exit(1)
    log.info("[Step 2] Trained %s on %s training rows.", cfg.model_name, len(X_train_t))

    # ---- Step 3: evaluate on VALIDATION ONLY (never train, per the pitfall) ----
    try:
        val_proba = model.predict_proba(X_val_t)[:, 1]
        val_pred = model.predict(X_val_t)
        model_metrics = compute_metrics(y_val, val_proba, val_pred, cfg.metrics)
    except Exception as e:
        log.error("Evaluation stage failed: %s", e)
        sys.exit(1)
    log.info("[Step 3] %s validation metrics: %s", cfg.model_name, model_metrics)

    # ---- Step 4: compare against baseline — is it ACTUALLY better? ----
    primary = cfg.primary_metric
    lift = model_metrics[primary] - baseline_metrics[primary]
    beats_baseline = lift > 0
    log.info("[Step 4] Comparison on primary metric '%s': model=%s baseline=%s lift=%+.4f -> %s",
              primary, model_metrics[primary], baseline_metrics[primary], lift,
              "BEATS baseline" if beats_baseline else "DOES NOT beat baseline")

    # ---- Step 5: inspect worst errors for patterns ----
    errors_df = worst_errors(X_val, y_val, val_pred, val_proba, top_n=cfg.worst_errors_to_inspect)
    error_summary = summarize_error_patterns(errors_df)
    log.info("[Step 5] Worst-error summary: %s", error_summary)

    # ---- Step 6: record the run + decide next improvement ----
    model_row = log_run(cfg.experiment_log_path, "first_model", cfg, model_metrics, split_sizes,
                         extra={"model_name": cfg.model_name,
                                "beats_baseline": beats_baseline,
                                "lift_over_baseline": round(float(lift), 4)})

    if not beats_baseline:
        next_step = "NO-GO: model does not beat baseline — revisit features/model choice before proceeding."
    elif error_summary.get("n_errors_inspected", 0) == 0:
        next_step = ("GO: model beats baseline with zero validation errors. Next: re-validate on the held-out "
                      "test set and a larger/harder dataset slice before trusting this fully — a perfect "
                      "validation score on a small, well-separated set can still be optimistic.")
    elif error_summary.get("false_negatives", 0) > error_summary.get("false_positives", 0):
        next_step = ("GO, but next improvement: false negatives dominate the errors — raise recall via class "
                      "weighting or a lower decision threshold, since a missed positive is the costlier mistake here.")
    else:
        next_step = "GO: model beats baseline. Next improvement: try a stronger model (e.g. tree ensemble) or more features."
    log.info("[Step 6] Next-improvement decision: %s", next_step)

    import joblib
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.model_dir / f"{cfg.model_name}.joblib")

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "seed": cfg.seed,
        "split_sizes": split_sizes,
        "baseline": {"strategy": cfg.baseline_strategy, "metrics": baseline_metrics},
        "model": {"name": cfg.model_name, "metrics": model_metrics},
        "primary_metric": primary,
        "lift_over_baseline": round(float(lift), 4),
        "beats_baseline": beats_baseline,
        "worst_errors_summary": error_summary,
        "next_improvement_decision": next_step,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (cfg.report_dir / "evaluation_report.json").write_text(json.dumps(result, indent=2))
    errors_df.to_csv(cfg.report_dir / "worst_errors.csv", index=False)

    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    (cfg.log_dir / "run_experiment.log").write_text(json.dumps(result, indent=2))

    log.info("Done in %ss. Report -> %s", result["runtime_seconds"], cfg.report_dir / "evaluation_report.json")
    return result


if __name__ == "__main__":
    print(main())
