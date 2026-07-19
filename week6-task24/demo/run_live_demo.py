"""
Live, end-to-end demo. Run this and read the console output -- every number
below comes from an actual run, not a claim in a doc. This is what "ask
them to walk you through one real example end-to-end" looks like for the
AI/ML slice.
"""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import TASK21_AUDIT_PATH, ML_GO_AHEAD_PATH, METRICS_REPORT_PATH, CANDIDATES_PATH
import data_generator
import task21_audit
import sign_off
import explain


def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    line("STAGE 0 -- Generate real-shaped, at-scale, messy candidate data")
    df = data_generator.generate()
    df.to_csv(CANDIDATES_PATH, index=False)
    print(f"Rows: {len(df)}  |  Jobs: {df['job_id'].nunique()}  |  "
          f"Duplicates present: {df.duplicated().sum()}  |  "
          f"Missing jd_match: {df['jd_match'].isna().sum()}  |  "
          f"Missing portfolio_score: {df['portfolio_score'].isna().sum()}")

    line("STAGE 1 -- Task 21 upstream dependency: baseline fairness audit")
    t21 = task21_audit.run()
    print(f"Finding: {t21['finding']}")
    print(f"Baseline disparate impact: {t21['disparate_impact']:.3f} "
          f"(target >= 0.80)")
    print(f"Group positive rates: {t21['group_positive_rates']}")
    print(f"Baseline quality (misleadingly high accuracy despite bias): "
          f"{t21['quality_metrics']}")

    line("STAGE 2 -- Task 24: full sign-off pipeline (loads Stage 1 as a dependency)")
    artifact = sign_off.run()
    print(f"Data quality handling: {json.dumps(artifact['data_quality'], indent=2)}")
    print(f"\nDumb baseline (tier-blind rule) disparate impact: "
          f"{artifact['baseline']['disparate_impact']:.3f}  "
          f"quality: {artifact['baseline']['quality']}")
    print(f"\nFairness ceiling (best any merit-only model could do): "
          f"{artifact['audit_trail']['fairness_ceiling_disparate_impact']:.3f}")
    print(f"\nMitigated model disparate impact: "
          f"{artifact['audit_trail']['mitigated_disparate_impact']:.3f}")
    print(f"Mitigated model quality vs merit ground truth (overall): "
          f"{artifact['mitigated_model_quality']['vs_merit_ground_truth']['overall']}")

    line("STAGE 3 -- Sign-off decision")
    print(f"DECISION: {artifact['decision']}")
    print(f"RATIONALE: {artifact['rationale']}")

    line("STAGE 4 -- One-example explainability walkthrough")
    explain.demo()

    line("STAGE 5 -- Failure-path check: what happens if the dependency is missing")
    try:
        sign_off.run(task21_path="/tmp/nonexistent_audit.json")
    except Exception as e:
        print(f"Correctly blocked: {type(e).__name__}: {e}")

    line("ARTIFACTS WRITTEN")
    print(f"- {TASK21_AUDIT_PATH}")
    print(f"- {ML_GO_AHEAD_PATH}")
    print(f"- {METRICS_REPORT_PATH}")


if __name__ == "__main__":
    main()
