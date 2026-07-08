# PlaceMux Rec Validation (Week 5 Task 20)

**Final Verdict: GO**

This repository contains the out-of-sample validation, distribution drift analysis, and scripted dry-run of the Rec v1 system. It produces a final data-driven Go/No-Go verdict.

## Key Findings

- **Model Performance**: The Rec v1 model achieved a Precision@5 of **0.5003** on fresh data, compared to the skill-overlap baseline's **0.4784**.
- **Distribution Drift**: A trained drift classifier reported an AUC of **0.5554** (minor drift).
- **Dry Run**: Scripted journeys correctly verified data isolation (3/3) and handled edge cases (3/3).

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
- `GET /college/{college_id}/recommendations` — Scoped recommendations list
