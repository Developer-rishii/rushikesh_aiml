"""
run_calibrated_classifier.py — Task 12's full flow, in the study guide's
exact step order:
  1. Train and tune the classifier on the full pipeline.
  2. Calibrate probabilities and verify with a calibration curve.
  3. Pick the cost-optimal threshold.
  4. Evaluate across folds and key segments for stability/fairness.
  5. Document the operating point and expected error rates.
  6. Package the model for serving.

Run: python -m src.run_calibrated_classifier
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from configs.loader import load_config
from src.data.dataset import load_dataframe, split_dataframe
from src.models.build import build_base_pipeline
from src.models.calibrate import fit_calibrated_variants, evaluate_calibration_quality, plot_calibration_curve
from src.evaluation.threshold import select_cost_optimal_threshold, confusion_at_threshold
from src.evaluation.stability import evaluate_segments
from src.evaluation.metrics import compute_metrics
from src.serving.package import package_for_serving, load_serving_package, predict as serving_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.run_calibrated_classifier")


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

    # ---- Step 1: train the base classifier (Task 9's confirmed config) ----
    try:
        base_pipeline = build_base_pipeline(cfg)
        base_pipeline.fit(X_train, y_train)
    except ValueError as e:
        log.error("Base model stage failed: %s", e)
        sys.exit(1)
    log.info("[Step 1] Trained base %s pipeline on %s rows.", cfg.model_name, len(X_train))

    # ---- Step 2: calibrate + verify ----
    try:
        variants = fit_calibrated_variants(base_pipeline, X_train, y_train, cfg)
        brier_scores = evaluate_calibration_quality(variants, base_pipeline, X_val, y_val)
        curve_path = plot_calibration_curve(variants, base_pipeline, X_val, y_val,
                                             cfg.n_calibration_bins, cfg.figure_dir)
    except Exception as e:
        log.error("Calibration stage failed: %s", e)
        sys.exit(1)

    best_method = min([m for m in brier_scores if m != "uncalibrated"], key=lambda m: brier_scores[m])
    calibrated_model = variants[best_method]
    log.info("[Step 2] Selected calibration method: %s (Brier=%.5f, vs uncalibrated=%.5f)",
              best_method, brier_scores[best_method], brier_scores["uncalibrated"])

    # ---- Step 3: cost-optimal threshold, on the CALIBRATED probabilities ----
    val_proba = calibrated_model.predict_proba(X_val)[:, 1]
    threshold_result, sweep = select_cost_optimal_threshold(y_val, val_proba, cfg)
    chosen_threshold = threshold_result["recommended_threshold"]["threshold"]

    # ---- Step 4: CV stability (train-only folds) + segment fairness (validation) ----
    cv = StratifiedKFold(n_splits=cfg.eval_cv_folds, shuffle=True, random_state=cfg.seed)
    fold_scores = []
    for tr_idx, te_idx in cv.split(X_train, y_train):
        fold_base = build_base_pipeline(cfg)
        fold_base.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        fold_calibrated = fit_calibrated_variants(
            fold_base, X_train.iloc[tr_idx], y_train.iloc[tr_idx], cfg
        )[best_method]
        fold_proba = fold_calibrated.predict_proba(X_train.iloc[te_idx])[:, 1]
        fold_scores.append(average_precision_score(y_train.iloc[te_idx], fold_proba))
    fold_scores = np.array(fold_scores)
    cv_stability = {
        "cv_folds": cfg.eval_cv_folds,
        "fold_scores": [round(float(s), 4) for s in fold_scores],
        "mean": round(float(fold_scores.mean()), 4),
        "std": round(float(fold_scores.std()), 4),
        "stable": bool(fold_scores.std() < 0.05),
    }
    log.info("[Step 4] CV stability: %s", cv_stability)

    try:
        segment_result = evaluate_segments(X_val, y_val, val_proba, chosen_threshold, cfg)
    except ValueError as e:
        log.error("Segment evaluation failed: %s", e)
        sys.exit(1)

    # ---- Step 5: document operating point + expected error rates (test set, touched once) ----
    test_proba = calibrated_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= chosen_threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_proba, test_pred, cfg.metrics)
    test_confusion = confusion_at_threshold(y_test, test_proba, chosen_threshold)
    n_test = len(y_test)
    operating_point = {
        "calibration_method": best_method,
        "threshold": chosen_threshold,
        "test_metrics": test_metrics,
        "test_confusion_matrix": test_confusion,
        "expected_missed_malignancy_rate": round(test_confusion["missed_malignancy"] / n_test, 4),
        "expected_unnecessary_biopsy_rate": round(test_confusion["unnecessary_biopsy"] / n_test, 4),
    }
    log.info("[Step 5] Operating point documented: %s", operating_point)

    # ---- Step 6: package for serving ----
    metadata = {
        "seed": cfg.seed,
        "model_name": cfg.model_name,
        "calibration_method": best_method,
        "feature_names": list(X_train.columns),
        "brier_scores": brier_scores,
        "cv_stability": cv_stability,
        "segment_check": segment_result,
        "operating_point": operating_point,
    }
    package_paths = package_for_serving(calibrated_model, chosen_threshold, metadata, cfg.serving_dir)

    # ---- verify the packaged artifact actually reproduces the same predictions ----
    reloaded_model, reloaded_config = load_serving_package(cfg.serving_dir)
    reloaded_proba, reloaded_decision = serving_predict(reloaded_model, reloaded_config, X_test)
    serving_matches = bool((abs(reloaded_proba - test_proba) < 1e-9).all())
    log.info("[Step 6] Serving package reload check: predictions match original = %s", serving_matches)

    # ---- persist run report ----
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "seed": cfg.seed,
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "brier_scores": brier_scores,
        "selected_calibration_method": best_method,
        "calibration_curve_path": curve_path,
        "threshold_selection": threshold_result,
        "chosen_threshold": chosen_threshold,
        "cv_stability": cv_stability,
        "segment_check": segment_result,
        "operating_point": operating_point,
        "serving_package_paths": package_paths,
        "serving_reload_verified": serving_matches,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (cfg.report_dir / "calibrated_classifier_report.json").write_text(json.dumps(result, indent=2, default=str))
    (cfg.log_dir / "run_calibrated_classifier.log").write_text(json.dumps(result, indent=2, default=str))

    log.info("Done in %ss. Report -> %s", result["runtime_seconds"],
              cfg.report_dir / "calibrated_classifier_report.json")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
