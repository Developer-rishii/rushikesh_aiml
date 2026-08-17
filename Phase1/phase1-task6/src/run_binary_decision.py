"""
run_binary_decision.py — Task 6's full flow, in the study guide's exact
step order:
  1. Train a binary classifier on preprocessed data.
  2. Produce a confusion matrix on validation.
  3. Compute precision, recall, F1 — not just accuracy.
  4. Plot ROC/PR curves and pick a threshold for the business cost.
  5. Check behaviour under class imbalance.
  6. Recommend a threshold tied to the real cost of each error.

Run: python -m src.run_binary_decision
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
from src.modeling.registry import build_model
from src.evaluation.decision_metrics import confusion_matrix_at_threshold, decision_metrics_at_threshold
from src.evaluation.curves import compute_curves, plot_curves
from src.evaluation.threshold_selection import select_cost_optimal_threshold
from src.evaluation.imbalance_check import check_imbalance

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.run_binary_decision")


def main():
    t0 = time.time()
    cfg = load_config()
    log.info("Loaded config: %s", cfg)

    # ---- data + preprocessing ----
    try:
        raw = load_dataframe(cfg)
        enriched = enrich_dataframe(raw, cfg)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(enriched, cfg)
        preprocessor = fit_preprocessor(X_train, cfg)
        verify_no_leakage(preprocessor, X_train, X_val)
        X_train_t = transform(preprocessor, X_train)
        X_val_t = transform(preprocessor, X_val)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error("Data/preprocessing stage failed: %s", e)
        sys.exit(1)

    # ---- Step 1: train the binary classifier ----
    try:
        model = build_model(cfg.model_name, cfg.model_params)
        model.fit(X_train_t, y_train)
        y_proba = model.predict_proba(X_val_t)[:, 1]
    except Exception as e:
        log.error("Model training/prediction failed: %s", e)
        sys.exit(1)
    log.info("[Step 1] Trained %s on %s rows, produced validation probabilities.",
              cfg.model_name, len(X_train_t))

    # ---- Step 2: confusion matrix at default 0.5 ----
    cm_default = confusion_matrix_at_threshold(y_val, y_proba, 0.5)
    log.info("[Step 2] Confusion matrix @ 0.5: %s", cm_default)

    # ---- Step 3: precision/recall/F1, not just accuracy ----
    try:
        metrics_default = decision_metrics_at_threshold(y_val, y_proba, 0.5, cfg.metrics)
    except ValueError as e:
        log.error("Metrics computation failed: %s", e)
        sys.exit(1)
    log.info("[Step 3] Metrics @ 0.5 (precision/recall/F1 + AUCs, not just accuracy): %s", metrics_default)

    # ---- Step 4: ROC/PR curves ----
    curves = compute_curves(y_val, y_proba)
    curve_paths = plot_curves(curves, chosen_threshold=0.5, out_dir=cfg.figure_dir)
    log.info("[Step 4] ROC AUC=%s, PR AUC=%s, plots saved to %s",
              curves["roc"]["auc"], curves["pr"]["auc"], cfg.figure_dir)

    # ---- Step 5: imbalance check ----
    imbalance = check_imbalance(y_train, y_val)
    log.info("[Step 5] Imbalance check: %s", imbalance)

    # ---- Step 6: cost-justified threshold recommendation ----
    threshold_result, sweep_df = select_cost_optimal_threshold(y_val, y_proba, cfg)
    recommended_t = threshold_result["recommended_threshold"]["threshold"]
    cm_recommended = confusion_matrix_at_threshold(y_val, y_proba, recommended_t)
    metrics_recommended = decision_metrics_at_threshold(y_val, y_proba, recommended_t, cfg.metrics)
    log.info("[Step 6] Recommended threshold=%s | confusion matrix: %s | metrics: %s",
              recommended_t, cm_recommended, metrics_recommended)

    # ---- persist everything ----
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    import joblib
    joblib.dump(model, cfg.model_dir / f"{cfg.model_name}.joblib")
    save_preprocessor(preprocessor, cfg.model_dir / "fitted_preprocessor.joblib")
    sweep_df.to_csv(cfg.report_dir / "threshold_sweep.csv", index=False)

    result = {
        "seed": cfg.seed,
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "model_name": cfg.model_name,
        "confusion_matrix_at_0.5": cm_default,
        "metrics_at_0.5": metrics_default,
        "curve_aucs": {"roc_auc": curves["roc"]["auc"], "pr_auc": curves["pr"]["auc"]},
        "curve_plots": curve_paths,
        "imbalance_check": imbalance,
        "threshold_selection": threshold_result,
        "confusion_matrix_at_recommended_threshold": cm_recommended,
        "metrics_at_recommended_threshold": metrics_recommended,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (cfg.report_dir / "binary_decision_report.json").write_text(json.dumps(result, indent=2))
    (cfg.log_dir / "run_binary_decision.log").write_text(json.dumps(result, indent=2))

    log.info("Done in %ss. Report -> %s", result["runtime_seconds"],
              cfg.report_dir / "binary_decision_report.json")
    return result


if __name__ == "__main__":
    print(main())
