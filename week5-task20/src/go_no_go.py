"""
Go/No-Go report generator: reads metrics.json, drift_results.json,
and dry_run_transcript.json to produce the final verdict.
"""
import json
import os


def generate_go_no_go(reports_dir: str) -> dict:
    """Generate the Go/No-Go verdict from actual pipeline results."""
    print("\n[GO/NO-GO] Generating verdict...")

    # Load all reports
    with open(os.path.join(reports_dir, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(reports_dir, "drift_results.json")) as f:
        drift = json.load(f)
    with open(os.path.join(reports_dir, "dry_run_transcript.json")) as f:
        dry_run = json.load(f)

    # ── Extract key numbers ──────────────────────────────────────────────
    fresh_model = metrics["fresh_data_model"]
    fresh_baseline = metrics["fresh_data_baseline"]
    orig_model = metrics.get("original_training_model", {})

    fresh_precision = fresh_model.get("precision_at_5", 0)
    baseline_precision = fresh_baseline.get("precision_at_5", 0)
    orig_precision = orig_model.get("precision_at_5", 0)

    precision_delta_vs_baseline = round(fresh_precision - baseline_precision, 4)
    precision_delta_vs_original = round(fresh_precision - orig_precision, 4)

    drift_auc = drift["drift_auc"]
    dry_run_all_passed = dry_run["all_passed"]
    isolation_checks = dry_run["isolation_checks"]
    failure_handled = dry_run["deliberate_failures_handled"]

    # ── Verdict logic ─────────────────────────────────────────────────────
    # GO if:
    # 1. Fresh model precision >= baseline precision (model adds value)
    # 2. Drift AUC < 0.75 (no severe drift)
    # 3. All dry-run checks pass
    conditions = {
        "model_beats_baseline": fresh_precision >= baseline_precision,
        "drift_acceptable": drift_auc < 0.75,
        "dry_run_passed": dry_run_all_passed,
    }

    verdict = "GO" if all(conditions.values()) else "NO-GO"

    # ── Build report ──────────────────────────────────────────────────────
    go_no_go = {
        "verdict": verdict,
        "thresholds": {
            "model_beats_baseline": f"fresh precision ({fresh_precision}) >= baseline ({baseline_precision})",
            "drift_acceptable": f"drift AUC ({drift_auc}) < 0.75",
            "dry_run_passed": f"all dry-run steps passed: {dry_run_all_passed}",
        },
        "conditions_met": conditions,
        "metrics_summary": {
            "fresh_model_precision_at_5": fresh_precision,
            "fresh_baseline_precision_at_5": baseline_precision,
            "original_model_precision_at_5": orig_precision,
            "precision_delta_vs_baseline": precision_delta_vs_baseline,
            "precision_delta_vs_original": precision_delta_vs_original,
            "fresh_model_recall_at_5": fresh_model.get("recall_at_5", 0),
            "fresh_model_mrr": fresh_model.get("mrr", 0),
            "fresh_model_fpr_at_5": fresh_model.get("fpr_at_5", 0),
        },
        "drift_summary": {
            "auc": drift_auc,
            "severity": drift["drift_severity"],
            "interpretation": drift["interpretation"],
            "drifted_features": drift["drifted_features"],
        },
        "dry_run_summary": {
            "total_steps": dry_run["total_steps"],
            "passed": dry_run["passed"],
            "failed": dry_run["failed"],
            "isolation_checks": isolation_checks,
            "deliberate_failures_handled": failure_handled,
        },
    }

    # Save JSON
    with open(os.path.join(reports_dir, "go_no_go.json"), "w") as f:
        json.dump(go_no_go, f, indent=2)

    # ── Generate markdown report ──────────────────────────────────────────
    md = f"""# PlaceMux Rec Validation — Go/No-Go Report

**Verdict: {verdict}**

## Decision Criteria

| Criterion | Threshold | Current | Met? |
|---|---|---|---|
| Model beats baseline | Fresh P@5 ≥ Baseline P@5 | {fresh_precision} vs {baseline_precision} | {'✅' if conditions['model_beats_baseline'] else '❌'} |
| Drift acceptable | Drift AUC < 0.75 | {drift_auc} | {'✅' if conditions['drift_acceptable'] else '❌'} |
| Dry run passed | All steps pass | {dry_run['passed']}/{dry_run['total_steps']} | {'✅' if conditions['dry_run_passed'] else '❌'} |

## Out-of-Sample Validation

| Metric | Original Model | Fresh Baseline | Fresh Model | Δ vs Baseline | Δ vs Original |
|---|---|---|---|---|---|
| Precision@5 | {orig_precision} | {baseline_precision} | {fresh_precision} | {precision_delta_vs_baseline:+.4f} | {precision_delta_vs_original:+.4f} |
| Recall@5 | {orig_model.get('recall_at_5', 'N/A')} | {fresh_baseline.get('recall_at_5', 'N/A')} | {fresh_model.get('recall_at_5', 'N/A')} | | |
| MRR | {orig_model.get('mrr', 'N/A')} | {fresh_baseline.get('mrr', 'N/A')} | {fresh_model.get('mrr', 'N/A')} | | |
| FPR@5 | {orig_model.get('fpr_at_5', 'N/A')} | {fresh_baseline.get('fpr_at_5', 'N/A')} | {fresh_model.get('fpr_at_5', 'N/A')} | | |

## Drift Detection

- **AUC**: {drift_auc}
- **Severity**: {drift['drift_severity']}
- **Interpretation**: {drift['interpretation']}
- **Drifted features (KS-test p < 0.05)**: {', '.join(drift['drifted_features']) if drift['drifted_features'] else 'None'}

## Dry Run Summary

- **Total steps**: {dry_run['total_steps']}
- **Passed**: {dry_run['passed']}
- **Failed**: {dry_run['failed']}
- **Isolation checks**: {isolation_checks}
- **Deliberate failures handled**: {failure_handled}
"""

    with open(os.path.join(reports_dir, "go_no_go_report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[GO/NO-GO] Verdict: {verdict}")
    print(f"[GO/NO-GO] Report saved to reports/go_no_go_report.md")

    return go_no_go


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    generate_go_no_go(os.path.join(base, "reports"))
