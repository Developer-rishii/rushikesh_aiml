"""
models/calibrate.py — Step 2: calibrate probabilities and verify with a
calibration curve. Both Platt (sigmoid) and isotonic are actually fit and
compared via Brier score (lower = better-calibrated), not assumed —
directly guards against the pitfall "uncalibrated probabilities used as
if exact" by making calibration quality a measured, reported number.
"""
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

log = logging.getLogger("src.models.calibrate")


def fit_calibrated_variants(base_pipeline, X_train, y_train, cfg) -> dict:
    variants = {}
    for method in cfg.calibration_methods:
        calibrated = CalibratedClassifierCV(base_pipeline, method=method, cv=cfg.calibration_cv_folds)
        calibrated.fit(X_train, y_train)
        variants[method] = calibrated
        log.info("[Step 2] Fit %s-calibrated classifier (internal cv=%s)", method, cfg.calibration_cv_folds)
    return variants


def evaluate_calibration_quality(variants: dict, uncalibrated_pipeline, X_val, y_val) -> dict:
    """Brier score for the raw model and each calibrated variant — lower
    is better-calibrated. This is the evidence for which method to pick,
    not a coin flip between Platt and isotonic."""
    results = {}
    raw_proba = uncalibrated_pipeline.predict_proba(X_val)[:, 1]
    results["uncalibrated"] = round(float(brier_score_loss(y_val, raw_proba)), 5)
    for method, model in variants.items():
        proba = model.predict_proba(X_val)[:, 1]
        results[method] = round(float(brier_score_loss(y_val, proba)), 5)
    log.info("[Step 2] Brier scores (lower=better-calibrated): %s", results)
    return results


def plot_calibration_curve(variants: dict, uncalibrated_pipeline, X_val, y_val, n_bins: int, out_dir: Path) -> str:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")

    raw_proba = uncalibrated_pipeline.predict_proba(X_val)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_val, raw_proba, n_bins=n_bins, strategy="uniform")
    ax.plot(mean_pred, frac_pos, marker="o", label="uncalibrated")

    for method, model in variants.items():
        proba = model.predict_proba(X_val)[:, 1]
        frac_pos, mean_pred = calibration_curve(y_val, proba, n_bins=n_bins, strategy="uniform")
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{method}-calibrated")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual)")
    ax.set_title("Calibration Curve — validation set")
    ax.legend()
    path = out_dir / "calibration_curve.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("[Step 2] Saved calibration curve -> %s", path)
    return str(path)
