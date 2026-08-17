"""
run_feature_engineering.py — Task 7's full flow, in the study guide's
exact step order:
  1. Re-confirm the target definition and label quality.
  2. Derive candidate features from domain reasoning.
  3. Add aggregate/time-based features where relevant.
  4. Train a model and inspect feature importance.
  5. Prune useless/leaky features.
  6. Lock a baseline feature set for the pipeline.

Run: python -m src.run_feature_engineering
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from configs.loader import load_config
from src.data.dataset import load_dataframe, split_dataframe
from src.features.target_check import confirm_target
from src.features.engineer import derive_domain_features, derive_aggregate_features
from src.features.importance import compute_permutation_importance
from src.features.pruning import check_leakage, prune_by_importance
from src.preprocessing.simple import fit_transform_train, transform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.run_feature_engineering")


def main():
    t0 = time.time()
    cfg = load_config()
    log.info("Loaded config: %s", cfg)

    # ---- Step 1: re-confirm target ----
    try:
        raw = load_dataframe(cfg)
        target_check = confirm_target(raw, cfg.target_col)
    except (FileNotFoundError, ValueError) as e:
        log.error("Target confirmation failed: %s", e)
        sys.exit(1)

    original_feature_cols = [c for c in raw.columns if c != cfg.target_col]

    # ---- Steps 2/3: derive candidate features (on the FULL df, still
    # row-wise-only, before the split — this is safe because every
    # formula uses only same-row inputs, never cross-row statistics
    # like a global mean; verified explicitly in tests) ----
    try:
        enriched, domain_added = derive_domain_features(raw)
        enriched, agg_added = derive_aggregate_features(enriched)
    except Exception as e:
        log.error("Feature engineering failed: %s", e)
        sys.exit(1)
    candidate_new_features = domain_added + agg_added
    log.info("Candidate new features (%s total): %s", len(candidate_new_features), candidate_new_features)

    # ---- Step 5a: leakage gate on candidates BEFORE they ever reach the model ----
    try:
        leaky = check_leakage(enriched, cfg.target_col, candidate_new_features, cfg.leakage_corr_threshold)
    except Exception as e:
        log.error("Leakage check failed: %s", e)
        sys.exit(1)
    candidate_new_features = [c for c in candidate_new_features if c not in leaky]

    all_feature_cols = original_feature_cols + candidate_new_features

    # ---- split (on the enriched df, so new features get split consistently) ----
    try:
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataframe(
            enriched[all_feature_cols + [cfg.target_col]], cfg
        )
    except (ValueError, RuntimeError) as e:
        log.error("Split failed: %s", e)
        sys.exit(1)

    if X_train.isna().all(axis=0).any():
        bad = X_train.columns[X_train.isna().all(axis=0)].tolist()
        log.error("Column(s) entirely missing in training data: %s", bad)
        sys.exit(1)

    # ---- preprocess (fit on train only) ----
    X_train_p, imputer, scaler = fit_transform_train(X_train, cfg)
    X_val_p = transform(X_val, imputer, scaler)

    # ---- Step 4: train + importance ----
    try:
        model = LogisticRegression(**cfg.model_params)
        model.fit(X_train_p, y_train)
        val_proba = model.predict_proba(X_val_p)[:, 1]
        pr_auc_all_features = round(float(average_precision_score(y_val, val_proba)), 4)
        importance_df = compute_permutation_importance(model, X_val_p, y_val, cfg)
    except Exception as e:
        log.error("Model training/importance stage failed: %s", e)
        sys.exit(1)
    log.info("[Step 4] PR-AUC with all %s candidate features: %s", len(all_feature_cols), pr_auc_all_features)

    # ---- Step 5b: usefulness gate (only on NEW engineered features; the
    # original Task 2 baseline is protected from this specific gate) ----
    prune_result = prune_by_importance(
        importance_df, cfg.min_importance_lift, protected=set(original_feature_cols)
    )
    final_features = [f for f in all_feature_cols if f in prune_result["kept"]]

    # ---- retrain on the FINAL pruned set, to measure the actual lift ----
    X_train_final = X_train_p[final_features]
    X_val_final = X_val_p[final_features]
    model_final = LogisticRegression(**cfg.model_params)
    model_final.fit(X_train_final, y_train)
    val_proba_final = model_final.predict_proba(X_val_final)[:, 1]
    pr_auc_final = round(float(average_precision_score(y_val, val_proba_final)), 4)

    # also measure PR-AUC with the ORIGINAL Task 2 baseline alone (no
    # engineered features at all) as the honest comparison point
    X_train_orig = X_train_p[original_feature_cols]
    X_val_orig = X_val_p[original_feature_cols]
    model_orig = LogisticRegression(**cfg.model_params)
    model_orig.fit(X_train_orig, y_train)
    pr_auc_original_only = round(float(average_precision_score(
        y_val, model_orig.predict_proba(X_val_orig)[:, 1])), 4)

    engineered_kept = [f for f in final_features if f in candidate_new_features]
    lift_vs_original = round(pr_auc_final - pr_auc_original_only, 4)

    log.info("[Step 6] PR-AUC original-only=%s | all-candidates=%s | final-pruned=%s | "
              "lift from kept engineered features=%+.4f",
              pr_auc_original_only, pr_auc_all_features, pr_auc_final, lift_vs_original)

    # ---- Step 6: lock the baseline feature set ----
    cfg.baseline_feature_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    baseline_lock = {
        "locked_at_seed": cfg.seed,
        "target_col": cfg.target_col,
        "final_feature_set": final_features,
        "n_original_features": len(original_feature_cols),
        "n_engineered_kept": len(engineered_kept),
        "engineered_kept": engineered_kept,
        "engineered_dropped_leakage": leaky,
        "engineered_dropped_low_importance": prune_result["dropped_low_importance"],
        "pr_auc_original_only": pr_auc_original_only,
        "pr_auc_final_locked_set": pr_auc_final,
        "lift_over_original_baseline": lift_vs_original,
    }
    (cfg.baseline_feature_dir / "locked_feature_set.json").write_text(json.dumps(baseline_lock, indent=2))
    importance_df.to_csv(cfg.report_dir / "feature_importance.csv", index=False)

    result = {
        "seed": cfg.seed,
        "target_check": target_check,
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "candidate_features_derived": len(domain_added) + len(agg_added),
        "leakage_dropped": leaky,
        "low_importance_dropped": prune_result["dropped_low_importance"],
        "baseline_lock": baseline_lock,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (cfg.report_dir / "feature_engineering_report.json").write_text(json.dumps(result, indent=2))
    (cfg.log_dir / "run_feature_engineering.log").write_text(json.dumps(result, indent=2))

    log.info("Done in %ss. Locked feature set (%s features) -> %s",
              result["runtime_seconds"], len(final_features),
              cfg.baseline_feature_dir / "locked_feature_set.json")
    return result


if __name__ == "__main__":
    print(main())
