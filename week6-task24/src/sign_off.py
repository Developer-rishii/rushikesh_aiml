"""
Task 24 deliverable: Fairness close + sign-off.

Full chain, each stage wrapped so a failure produces a clear, attributable
error rather than a stack trace the launch team can't act on:

  1. Load & validate the Task 21 upstream dependency (fail loudly if missing/malformed)
  2. Load & clean the candidate data (fail loudly if schema is broken)
  3. Run the dumb baseline (what "good" is measured against)
  4. Train the mitigated model (reweighing + threshold post-processing)
  5. Compute the fairness ceiling reference
  6. Apply sign-off decision logic
  7. Persist the ML go-ahead artifact + full metrics report
"""
import json
import hashlib
import datetime as dt
import sys

import pandas as pd

from config import (CANDIDATES_PATH, TASK21_AUDIT_PATH, ML_GO_AHEAD_PATH,
                     METRICS_REPORT_PATH, PROTECTED_ATTR, DI_TARGET,
                     CEILING_TOLERANCE, SEED)
from dependency_loader import load_task21_audit, DependencyError
from data_validation import clean, DataValidationError
from baseline_model import baseline_predict
from mitigate import train_mitigated
from audit import fairness_ceiling
from metrics import disparate_impact, classification_report, segmented_report


def decide(mitigated_di, ceiling_di) -> tuple[str, str]:
    if mitigated_di is None:
        return "WITHHELD", "Disparate impact could not be computed (missing group data)."
    if mitigated_di >= DI_TARGET:
        return (
            "SIGNED_OFF",
            f"Mitigated disparate impact {mitigated_di:.3f} meets the 4/5ths "
            f"rule threshold of {DI_TARGET:.2f}.",
        )
    if ceiling_di is not None and mitigated_di >= CEILING_TOLERANCE * ceiling_di:
        return (
            "CONDITIONALLY_SIGNED_OFF",
            f"Mitigated disparate impact {mitigated_di:.3f} is below the "
            f"{DI_TARGET:.2f} target but within {int((1 - CEILING_TOLERANCE) * 100)}% "
            f"of the {ceiling_di:.3f} fairness ceiling measured on this dataset. "
            "Sign-off is conditional on re-auditing once more production data "
            "is collected.",
        )
    return (
        "WITHHELD",
        f"Mitigated disparate impact {mitigated_di:.3f} is below both the "
        f"{DI_TARGET:.2f} target and the fairness ceiling "
        f"({ceiling_di if ceiling_di is not None else 'n/a'}). Do not launch "
        "on this model.",
    )


def run(seed: int = SEED, candidates_path: str = CANDIDATES_PATH,
        task21_path: str = TASK21_AUDIT_PATH, out_path: str = ML_GO_AHEAD_PATH,
        metrics_out_path: str = METRICS_REPORT_PATH) -> dict:

    # --- Stage 1: upstream dependency ---
    try:
        task21 = load_task21_audit(task21_path)
    except DependencyError as e:
        print(f"[BLOCKED] {e}", file=sys.stderr)
        raise

    # --- Stage 2: load & clean data ---
    try:
        raw_df = pd.read_csv(candidates_path)
    except FileNotFoundError as e:
        raise DataValidationError(
            f"Candidate data not found at '{candidates_path}'. Run data_generator.py first."
        ) from e
    clean_log: dict = {}
    df = clean(raw_df, report=clean_log)

    # --- Stage 3: dumb baseline (what "good" is judged against) ---
    baseline_preds = baseline_predict(df)
    baseline_quality = classification_report(df["historical_recommended"].values, baseline_preds)
    baseline_di = disparate_impact(baseline_preds, df[PROTECTED_ATTR].values)

    # --- Stage 4: mitigated model ---
    model, thresholds, split_data, mitigated_audit = train_mitigated(df, seed=seed)
    X_test, y_test_historical, tier_test, mitigated_preds, df_test = split_data
    y_test_fair = df_test["fair_recommended"].values

    # Quality against the historical (biased) label is EXPECTED to look worse
    # for previously over-favoured/under-favoured groups -- that divergence is
    # the mitigation working as intended, not a defect. Quality against the
    # merit ground truth (fair_recommended) is the number that actually
    # matters and is not achievable to compute in a real deployment (no
    # bias-free ground truth exists there), so we report both, labelled.
    mitigated_quality_vs_historical = segmented_report(y_test_historical, mitigated_preds, tier_test)
    mitigated_quality_vs_fair_ground_truth = segmented_report(y_test_fair, mitigated_preds, tier_test)

    # --- Stage 5: fairness ceiling reference ---
    ceiling_audit = fairness_ceiling(df, seed=seed)

    # --- Stage 6: decision ---
    decision, rationale = decide(mitigated_audit["disparate_impact"], ceiling_audit["disparate_impact"])

    data_fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]

    artifact = {
        "artifact": "ML_GO_AHEAD",
        "task": "Task 24 - Fairness close + sign-off",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_version": f"placemux-rec-v3-mitigated-{data_fingerprint}",
        "protected_attribute": PROTECTED_ATTR,
        "di_target": DI_TARGET,
        "upstream_dependency": {
            "source": "task21_audit_results.json",
            "task21_finding": task21["finding"],
            "task21_disparate_impact": task21["disparate_impact"],
        },
        "data_quality": {
            "n_rows_raw": len(raw_df),
            "n_rows_after_cleaning": clean_log["rows_after"],
            "duplicates_removed": clean_log["duplicates_removed"],
            "missing_values_imputed": clean_log["missing_values_imputed_with_median"],
            "outliers_capped": clean_log["years_exp_outliers_capped"],
        },
        "baseline": {
            "description": "dumb rule: recommend if 0.6*skill_score + 0.4*jd_match >= 60",
            "disparate_impact": baseline_di["disparate_impact"],
            "quality": baseline_quality,
        },
        "audit_trail": {
            "task21_baseline_disparate_impact": task21["disparate_impact"],
            "task21_baseline_group_rates": task21["group_positive_rates"],
            "fairness_ceiling_disparate_impact": ceiling_audit["disparate_impact"],
            "fairness_ceiling_group_rates": ceiling_audit["group_rates"],
            "mitigated_disparate_impact": mitigated_audit["disparate_impact"],
            "mitigated_group_rates": mitigated_audit["group_rates"],
        },
        "mitigated_model_quality": {
            "note": (
                "Quality vs the historical label is expected to look worse for "
                "tier 1 (previously over-recommended) and tier 3 (previously "
                "under-recommended) -- that gap is the bias correction working "
                "as intended, not a defect. Quality vs the merit ground truth "
                "(fair_recommended, a synthetic-data-only reference not "
                "available in real deployments) is the number that reflects "
                "actual correctness."
            ),
            "vs_historical_biased_label": mitigated_quality_vs_historical,
            "vs_merit_ground_truth": mitigated_quality_vs_fair_ground_truth,
        },
        "mitigation_method": [
            "Kamiran-Calders reweighing on training samples (group x label independence)",
            "GradientBoostingClassifier (150 trees, depth 3) replacing plain logistic regression",
            "Protected attribute (college_tier) dropped from model features",
            "Per-group decision-threshold post-processing to equalise positive rates",
        ],
        "per_group_thresholds": {str(k): v for k, v in thresholds.items()},
        "decision": decision,
        "rationale": rationale,
        "open_items": [
            "Re-audit quarterly once real production outcome data replaces synthetic data.",
            "This sign-off covers the AI/ML recommendation model only; data-deletion "
            "(security) and load-test (backend) items are owned by other tracks.",
        ],
    }

    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    with open(metrics_out_path, "w") as f:
        json.dump({
            "baseline_quality": baseline_quality,
            "mitigated_quality_vs_historical_biased_label": mitigated_quality_vs_historical,
            "mitigated_quality_vs_merit_ground_truth": mitigated_quality_vs_fair_ground_truth,
            "disparate_impact": {
                "task21_baseline": task21["disparate_impact"],
                "rule_based_baseline": baseline_di["disparate_impact"],
                "mitigated": mitigated_audit["disparate_impact"],
                "fairness_ceiling": ceiling_audit["disparate_impact"],
            },
        }, f, indent=2)

    return artifact


if __name__ == "__main__":
    artifact = run()
    print(json.dumps(artifact, indent=2))
