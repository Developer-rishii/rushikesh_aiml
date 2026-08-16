"""
run_protocol.py — runs the full pre-processing protocol end-to-end and
demonstrates BOTH train-time fitting and inference-time reuse of the same
saved artifact, which is the actual Definition of Done for this task
("reusable at train and inference time" — not just fit once).

Run: python -m src.run_protocol
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.data.dataset import load_dataframe, enrich_dataframe, split_dataframe
from src.preprocessing.pipeline import (
    list_feature_types, fit_preprocessor, transform,
    verify_no_leakage, save_preprocessor, load_preprocessor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.run_protocol")


def main():
    t0 = time.time()
    cfg = load_config()
    log.info("Loaded config: %s", cfg)

    # ---- load + enrich + split ----
    try:
        raw = load_dataframe(cfg)
        enriched = enrich_dataframe(raw, cfg)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(enriched, cfg)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        log.error("Data stage failed: %s", e)
        sys.exit(1)

    feature_types = list_feature_types(X_train)
    log.info("Step 1 — feature inventory: %s numeric, %s categorical",
              len(feature_types["numeric"]), len(feature_types["categorical"]))
    missing_before = int(X_train.isna().sum().sum())
    log.info("Missing values present in training data (must be imputed, not dropped): %s",
              missing_before)

    # ---- fit ONLY on train (steps 2-4) ----
    try:
        preprocessor = fit_preprocessor(X_train, cfg)
    except ValueError as e:
        log.error("Preprocessor fit failed: %s", e)
        sys.exit(1)

    # ---- transform each split with the SAME fitted object ----
    try:
        X_train_t = transform(preprocessor, X_train)
        X_val_t = transform(preprocessor, X_val)
        X_test_t = transform(preprocessor, X_test)
    except ValueError as e:
        log.error("Transform failed: %s", e)
        sys.exit(1)

    rows_dropped = len(X_train) - X_train_t.shape[0]
    if rows_dropped != 0:
        log.error("Row count changed during preprocessing (%s rows lost) — "
                   "rows must never be dropped for missing values.", rows_dropped)
        sys.exit(1)
    log.info("Confirmed zero rows dropped: input=%s, output=%s (missing values were "
              "imputed, not removed)", len(X_train), X_train_t.shape[0])

    assert not (X_train_t != X_train_t).any(), "NaNs remained in transformed train output"
    log.info("Confirmed zero NaNs remain after transform (all %s missing values imputed).",
              missing_before)

    # ---- Step 5: verify no leakage ----
    try:
        leakage_result = verify_no_leakage(preprocessor, X_train, X_val)
    except RuntimeError as e:
        log.error("LEAKAGE DETECTED: %s", e)
        sys.exit(1)

    # ---- Step 6: save for inference reuse ----
    preprocessor_path = cfg.preprocessor_dir / "fitted_preprocessor.joblib"
    save_preprocessor(preprocessor, preprocessor_path)

    # ---- demonstrate INFERENCE-time reuse: load fresh, transform new rows ----
    reloaded = load_preprocessor(preprocessor_path)
    # simulate "new data arriving at inference time" using held-out test rows
    inference_batch = X_test.iloc[:5]
    inference_output = transform(reloaded, inference_batch)
    log.info("Inference-time reuse check: transformed %s new rows with the "
              "reloaded artifact, output shape=%s (no re-fitting occurred).",
              len(inference_batch), inference_output.shape)

    # sanity: reloaded preprocessor gives identical output to the in-memory one
    import numpy as np
    same_output = np.allclose(
        transform(preprocessor, inference_batch), inference_output
    )
    if not same_output:
        log.error("Reloaded preprocessor output differs from original — train/serve drift detected.")
        sys.exit(1)
    log.info("Confirmed reloaded artifact output == original fitted object output "
              "(no train/serve drift).")

    # ---- write report + log ----
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "seed": cfg.seed,
        "rows": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "feature_types": feature_types,
        "missing_values_in_train_before_impute": missing_before,
        "rows_dropped_during_preprocessing": rows_dropped,
        "output_shape": {
            "train": list(X_train_t.shape), "val": list(X_val_t.shape), "test": list(X_test_t.shape),
        },
        "leakage_check": leakage_result,
        "preprocessor_path": str(preprocessor_path),
        "inference_reuse_verified": bool(same_output),
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (cfg.report_dir / "preprocessing_report.json").write_text(json.dumps(result, indent=2))
    (cfg.log_dir / "run_protocol.log").write_text(json.dumps(result, indent=2))
    log.info("Done in %ss. Report -> %s", result["runtime_seconds"],
              cfg.report_dir / "preprocessing_report.json")
    return result


if __name__ == "__main__":
    print(main())
