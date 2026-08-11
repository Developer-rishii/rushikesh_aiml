"""
Step 6 of the build pipeline: decide go/no-go and the modelling approach.

Also serves as the smoke test: runs the full chain (ingest -> enrich ->
profile -> leakage-check -> balance-check -> split -> baseline model ->
logged metrics) end-to-end on the real dataset carried over from Task 1,
with the same fixed seed, and writes a timestamped log of everything that
happened.

Run standalone: python src/run_pipeline.py
"""
import sys
import json
import time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, accuracy_score,
)

from configs.config import (
    SEED, LOGS_DIR, TARGET_COL, TRAIN_FRAC, VAL_FRAC, TEST_FRAC,
)
from data_ingestion import load_task1_raw, enrich_with_records_system_fields
from feature_profiling import profile_features
from leakage_check import run_leakage_check
from balance_report import run_balance_report
from configs.config import ENRICHED_PATH, CLEAN_PATH, REPORTS_DIR

np.random.seed(SEED)


def log(msg, log_lines):
    print(msg)
    log_lines.append(msg)


def build_model_pipeline():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ])


def main():
    log_lines = []
    t0 = time.time()
    log(f"=== Task 2 pipeline run | seed={SEED} | continuing from Task 1's raw data ===", log_lines)

    try:
        raw = load_task1_raw()
        enriched = enrich_with_records_system_fields(raw, seed=SEED)
        enriched.to_csv(ENRICHED_PATH, index=False)
        log(f"[1/6] Loaded Task 1 data ({raw.shape[0]} rows x {raw.shape[1]} cols), "
            f"enriched with records-system fields -> {enriched.shape[1]} cols", log_lines)
    except Exception as e:
        log(f"FATAL: data ingestion failed: {e}", log_lines)
        sys.exit(1)

    if enriched[TARGET_COL].isna().any():
        log("FATAL: target column contains missing values; cannot proceed.", log_lines)
        sys.exit(1)

    profile = profile_features(enriched, TARGET_COL)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    profile.to_csv(REPORTS_DIR / "feature_profile.csv", index=False)
    log(f"[2/6] Profiled {len(profile)} candidate features.", log_lines)

    clean_df, stat_suspects, dropped = run_leakage_check(enriched, TARGET_COL)
    clean_df.to_csv(CLEAN_PATH, index=False)
    log(f"[3/6] Leakage check dropped {len(dropped)} feature(s): {sorted(dropped)}", log_lines)

    counts, rates, majority_rate = run_balance_report(clean_df, TARGET_COL)
    log(f"[4/6] Class balance -> {dict(counts)} | majority baseline={majority_rate:.2%}", log_lines)

    y = clean_df[TARGET_COL]
    X = clean_df.drop(columns=[TARGET_COL])

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_FRAC + TEST_FRAC), stratify=y, random_state=SEED
    )
    rel_test = TEST_FRAC / (VAL_FRAC + TEST_FRAC)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=SEED
    )
    log(f"[5/6] Split (stratified, seed={SEED}): "
        f"train={len(X_train)} val={len(X_val)} test={len(X_test)} "
        f"(fracs {TRAIN_FRAC}/{VAL_FRAC}/{TEST_FRAC})", log_lines)

    try:
        model = build_model_pipeline()
        model.fit(X_train, y_train)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_pred = model.predict(X_val)
        metrics = {
            "val_pr_auc": round(float(average_precision_score(y_val, val_proba)), 4),
            "val_roc_auc": round(float(roc_auc_score(y_val, val_proba)), 4),
            "val_precision": round(float(precision_score(y_val, val_pred)), 4),
            "val_recall": round(float(recall_score(y_val, val_pred)), 4),
            "val_f1": round(float(f1_score(y_val, val_pred)), 4),
            "val_accuracy": round(float(accuracy_score(y_val, val_pred)), 4),
            "majority_baseline_accuracy": round(float(majority_rate), 4),
        }
    except Exception as e:
        log(f"FATAL: model training/eval failed: {e}", log_lines)
        sys.exit(1)

    log(f"[6/6] Baseline logistic regression (class_weight=balanced) validation metrics: "
        f"{metrics}", log_lines)

    minority_rate = 1 - majority_rate
    beats_baseline = metrics["val_pr_auc"] > minority_rate
    go_no_go = "GO" if beats_baseline and metrics["val_recall"] > 0.85 else "NO-GO (needs iteration)"
    log(f"\nGO/NO-GO DECISION: {go_no_go}", log_lines)
    log(f"  Rationale: PR-AUC={metrics['val_pr_auc']} vs minority base rate "
        f"{minority_rate:.4f}; recall={metrics['val_recall']} "
        f"(high recall bar because a missed malignant case is the costly error).", log_lines)
    log("  Modelling approach recommendation: binary classification on the "
        "cleaned 30-feature WDBC set, PR-AUC as primary metric, class-weighted "
        "linear model as baseline (already strong on this well-separated real "
        "dataset); escalate to a tree ensemble only if recall needs to improve "
        "further.", log_lines)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "seed": SEED,
        "rows": int(enriched.shape[0]),
        "features_dropped_leakage": sorted(dropped),
        "class_balance": {str(k): int(v) for k, v in counts.items()},
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "metrics": metrics,
        "go_no_go": go_no_go,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (LOGS_DIR / "run_metrics.json").write_text(json.dumps(result, indent=2))
    (LOGS_DIR / "run_log.txt").write_text("\n".join(log_lines))

    log(f"\nDone in {result['runtime_seconds']}s. "
        f"Logs -> {LOGS_DIR}/run_log.txt , {LOGS_DIR}/run_metrics.json", log_lines)
    return result


if __name__ == "__main__":
    main()
