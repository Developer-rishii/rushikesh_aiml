import os
import sys

# Ensure src modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.generate_fresh_data import generate_fresh_data
from src.validate_oos import run_oos_validation
from src.drift_detection import run_drift_detection
from src.dry_run import run_dry_run
from src.go_no_go import generate_go_no_go

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    reports_dir = os.path.join(base_dir, "reports")
    task16_dir = os.path.join(base_dir, "..", "week5-task16")

    print("=" * 60)
    print("Rec Validation & Go/No-Go Pipeline (Task 20)")
    print("=" * 60)

    # 1. Generate fresh validation data
    print("\n[1/6] Generating fresh validation data...")
    fresh_stats = generate_fresh_data(data_dir)

    # 2. Out-of-sample validation
    print("\n[2/6] Running out-of-sample validation on Rec v1...")
    metrics = run_oos_validation(data_dir, reports_dir, task16_dir)

    # 3. Drift detection
    print("\n[3/6] Running distribution drift detection...")
    drift_results = run_drift_detection(data_dir, reports_dir, task16_dir)

    # 4. Dry-run harness (API endpoints must be testable via TestClient)
    print("\n[4/6] Executing dry-run harness (college & admin journeys)...")
    dry_run_summary = run_dry_run(reports_dir)

    # 5. Go/No-Go verdict & report
    print("\n[5/6] Generating Go/No-Go verdict...")
    go_no_go = generate_go_no_go(reports_dir)

    # 6. Generate README.md dynamically
    print("\n[6/6] Auto-generating README.md...")
    readme_content = f"""# PlaceMux Rec Validation (Week 5 Task 20)

**Final Verdict: {go_no_go['verdict']}**

This repository contains the out-of-sample validation, distribution drift analysis, and scripted dry-run of the Rec v1 system. It produces a final data-driven Go/No-Go verdict.

## Key Findings

- **Model Performance**: The Rec v1 model achieved a Precision@5 of **{metrics['fresh_data_model'].get('precision_at_5', 'N/A')}** on fresh data, compared to the skill-overlap baseline's **{metrics['fresh_data_baseline'].get('precision_at_5', 'N/A')}**.
- **Distribution Drift**: A trained drift classifier reported an AUC of **{drift_results['drift_auc']}** ({drift_results['drift_severity']} drift).
- **Dry Run**: Scripted journeys correctly verified data isolation ({dry_run_summary['isolation_checks']}) and handled edge cases ({dry_run_summary['deliberate_failures_handled']}).

*(For full details, view `reports/go_no_go_report.md`.)*

## Quick Start

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the end-to-end validation pipeline (generates data, runs validation, executes dry-run, creates reports):
   ```bash
   python run_pipeline.py
   ```
3. Run the test suite:
   ```bash
   python -m pytest tests/ -v
   ```
4. Start the API:
   ```bash
   uvicorn src.api:app --reload
   ```

## Note for the Next Team (Go-ahead Handoff)

This validation suite is fully repeatable. When a new ranking model (e.g., Rec v2) is deployed to `week5-task16/src/models/ranker.joblib`, you can rerun `python run_pipeline.py` to instantly generate a fresh Go/No-Go report.

## API Endpoints

- `GET /validation/report` — Full out-of-sample validation numbers
- `GET /validation/drift` — Drift-detection result
- `GET /validation/dry-run` — Latest dry-run transcript summary
- `GET /validation/go-no-go` — Final verdict
- `GET /college/{{college_id}}/recommendations` — Scoped recommendations list
"""
    with open(os.path.join(base_dir, "README.md"), "w") as f:
        f.write(readme_content)

    print("\nPipeline complete!")

if __name__ == "__main__":
    main()
