"""
run_nonlinear.py — Task 10's full flow, in the study guide's exact step
order:
  1. Identify likely non-linear relationships from EDA.
  2. Train a more expressive model (gradient boosting).
  3. Compare validated performance to the linear baseline.
  4. Regularise/tune to control overfitting.
  5. Inspect partial dependence / feature effects for sense-checking.
  6. Keep the model only if the gain justifies the complexity.

Run: python -m src.run_nonlinear
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.data.dataset import load_dataframe, split_dataframe
from src.models.build import build_baseline_pipeline
from src.models.tune_nonlinear import tune_nonlinear_model
from src.evaluation.metrics import evaluate_pipeline
from src.evaluation.effects import top_features_by_importance, plot_partial_dependence

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.run_nonlinear")


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

    # ---- Step 1: EDA note on likely non-linearity (documented, not just asserted) ----
    eda_note = (
        "WDBC shape/size measurements (radius, area, concavity) are known to interact: "
        "a large-but-smooth tumor and a small-but-highly-concave tumor can carry similar "
        "risk through different mechanisms. A linear model can only add these effects; "
        "a tree ensemble can split on 'concavity high AND size moderate' style interactions "
        "the linear coefficients structurally cannot express — worth testing whether the "
        "data actually rewards that extra expressiveness."
    )
    log.info("[Step 1] %s", eda_note)

    # ---- baseline (linear) ----
    try:
        baseline_pipeline = build_baseline_pipeline(cfg)
        baseline_pipeline.fit(X_train, y_train)
        baseline_val_metrics = evaluate_pipeline(baseline_pipeline, X_val, y_val, cfg.metrics)
    except Exception as e:
        log.error("Baseline stage failed: %s", e)
        sys.exit(1)
    log.info("Linear baseline validation metrics: %s", baseline_val_metrics)

    # ---- Step 2/4: train + regularise the non-linear model (train-only CV) ----
    try:
        search, train_val_gap = tune_nonlinear_model(X_train, y_train, cfg)
        nonlinear_pipeline = search.best_estimator_
    except ValueError as e:
        log.error("Non-linear model stage failed: %s", e)
        sys.exit(1)
    except Exception as e:
        log.error("Non-linear tuning raised an unexpected error: %s", e)
        sys.exit(1)

    # ---- Step 3: compare validated performance to the linear baseline ----
    nonlinear_val_metrics = evaluate_pipeline(nonlinear_pipeline, X_val, y_val, cfg.metrics)
    primary = cfg.primary_metric
    val_lift = round(nonlinear_val_metrics[primary] - baseline_val_metrics[primary], 4)
    log.info("[Step 3] Validated comparison — baseline %s=%.4f, nonlinear %s=%.4f, lift=%+.4f",
              primary, baseline_val_metrics[primary], primary, nonlinear_val_metrics[primary], val_lift)

    overfit_flag = train_val_gap > 0.05
    log.info("[Step 4] Regularisation check — train-vs-CV gap for winning config=%.4f (%s)",
              train_val_gap, "POSSIBLE OVERFIT" if overfit_flag else "healthy")

    # ---- Step 5: partial dependence for sense-checking ----
    try:
        top_features = top_features_by_importance(nonlinear_pipeline, list(X_train.columns), cfg.pdp_n_top_features)
        pdp_path = plot_partial_dependence(nonlinear_pipeline, X_train, list(X_train.columns),
                                            top_features, cfg.figure_dir)
    except ValueError as e:
        log.error("Partial dependence stage failed: %s", e)
        sys.exit(1)

    # ---- Step 6: keep-or-reject decision, gated on validated (val-set) lift ----
    keep_nonlinear = val_lift >= cfg.min_lift_to_keep
    decision = (
        f"KEEP non-linear model: validated lift {val_lift:+.4f} >= threshold {cfg.min_lift_to_keep}"
        if keep_nonlinear else
        f"REJECT non-linear model, keep the linear baseline: validated lift {val_lift:+.4f} "
        f"does not clear the {cfg.min_lift_to_keep} threshold — the added complexity "
        f"(tree ensemble, tuned hyperparameters, lost coefficient-level interpretability) "
        f"is not justified by the evidence."
    )
    log.info("[Step 6] %s", decision)

    # ---- final test-set number for the record (whichever model is kept, evaluated once) ----
    kept_pipeline = nonlinear_pipeline if keep_nonlinear else baseline_pipeline
    kept_name = "gradient_boosting" if keep_nonlinear else "logreg_baseline"
    kept_test_metrics = evaluate_pipeline(kept_pipeline, X_test, y_test, cfg.metrics)
    log.info("Final kept model (%s) test-set metrics (touched once): %s", kept_name, kept_test_metrics)

    # ---- persist ----
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(baseline_pipeline, cfg.artifact_dir / "baseline_pipeline.joblib")
    joblib.dump(nonlinear_pipeline, cfg.artifact_dir / "nonlinear_pipeline.joblib")
    joblib.dump(kept_pipeline, cfg.artifact_dir / "kept_pipeline.joblib")

    result = {
        "seed": cfg.seed,
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "eda_note": eda_note,
        "baseline_val_metrics": baseline_val_metrics,
        "nonlinear_best_params": search.best_params_,
        "nonlinear_cv_score": round(float(search.best_score_), 4),
        "nonlinear_train_cv_gap": round(train_val_gap, 4),
        "possible_overfit": overfit_flag,
        "nonlinear_val_metrics": nonlinear_val_metrics,
        "primary_metric": primary,
        "validated_lift": val_lift,
        "min_lift_to_keep": cfg.min_lift_to_keep,
        "decision": decision,
        "kept_model": kept_name,
        "kept_model_test_metrics": kept_test_metrics,
        "pdp_top_features": top_features,
        "pdp_plot_path": pdp_path,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (cfg.report_dir / "nonlinear_report.json").write_text(json.dumps(result, indent=2, default=str))
    (cfg.log_dir / "run_nonlinear.log").write_text(json.dumps(result, indent=2, default=str))

    log.info("Done in %ss. Report -> %s", result["runtime_seconds"], cfg.report_dir / "nonlinear_report.json")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
