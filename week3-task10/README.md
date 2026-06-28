# Task 10 — Quality Sign-Off
AI/ML Engineer deliverable for Week 3, Phase 2 — Monetization Integration & Revenue Dashboard — PlaceMux, Altrodav Technologies.

This work is a regression-check extension of the existing matching logic into the monetization/payments domain. It confirms the matcher and reconciliation guardrails after paid applications and gateway events were introduced, rather than inventing a new matcher from scratch.

## Dependency / Scope
- `src/baseline.py`: existing matcher scoring logic used unchanged as the baseline comparator.
- `src/reconciliation.py`: newly built in this task to detect payment mismatches, failed/pending statuses, duplicate events, and charged-without-match cases.
- `src/evaluation.py`, `reports/generate_report.py`, `run_pipeline.py`, and `api/main.py`: newly built for Task 10 sign-off and live validation.
- `data/generate_dataset.py`: newly built synthetic monetization dataset generator for this task.
- `tests/test_reconciliation.py`: newly built regression and detection tests covering the actual edge cases.
- No prior task folder artifacts are imported into this repo; this README is based on the current built `week3-task10` repository state.

## Verdict
**Verdict: ⚠️ REGRESSION DETECTED — post-monetization recall dropped from 1.000 to 0.7778.**

The trained model beats the baseline on precision and FPR overall, but the post-monetization guardrail fails because recall degrades on paid applications.

## What’s actually in here
### Evaluation numbers from `reports/evaluation_results.json`
| Slice | Baseline P | Baseline R | Baseline FPR | Model P | Model R | Model FPR | N |
|---|---|---|---|---|---|---|---|
| Overall | 0.134 | 1.000 | 0.8235 | 0.6875 | 0.8462 | 0.0490 | 115 |
| Pre-monetization (free) | 0.1212 | 1.000 | 0.8056 | 0.5714 | 1.000 | 0.0833 | 40 |
| Post-monetization (paid basic/premium) | 0.1406 | 1.000 | 0.8333 | 0.7778 | 0.7778 | 0.0303 | 75 |

### Row counts evaluated
- `test_set_size`: 115 held-out application samples.
- `data/test_split.csv` provides the exact test application IDs used.
- No rows were dropped from the intended test split in `src/evaluation.py`; the evaluation uses the full held-out `test_split.csv` subset.

### Reconciliation verdict breakdown from `/signoff/reconciliation`
| Verdict bucket | Count |
|---|---|
| `amount_mismatches` | 23 |
| `failed_applications` | 107 |
| `duplicate_pairs` | 36 |
| `charged_without_match` | 84 |
| `needs_attention` | 107 |

### Key exact reconciliation metrics from live API
- `total_events`: 575
- `amount_mismatches`: 23
- `duplicate_pairs`: 36
- `charged_without_match`: 84
- `needs_attention`: 107

## Actual file tree (built repo state)
.
  demo.bat
  requirements.txt
  run_pipeline.py
  .pytest_cache
    .gitignore
    CACHEDIR.TAG
    README.md
    v
      cache
        nodeids
  api
    __init__.py
    main.py
    __pycache__
      __init__.cpython-311.pyc
      main.cpython-311.pyc
  data
    baseline_predictions.csv
    features_with_labels.csv
    generate_dataset.py
    jobs.csv
    monetization_events.csv
    students.csv
    test_split.csv
    __pycache__
      generate_dataset.cpython-311.pyc
  outputs
    Demo Match.png
    Evaluation Report.png
    Reconciliation Check.png
    localhost_8000_.png
  reports
    evaluation_results.json
    generate_report.py
    signoff_report.md
    __pycache__
      generate_report.cpython-311.pyc
  src
    __init__.py
    baseline.py
    evaluation.py
    explainability.py
    features.py
    labeling.py
    reconciliation.py
    train_model.py
    __pycache__
      __init__.cpython-311.pyc
      baseline.cpython-311.pyc
      evaluation.cpython-311.pyc
      explainability.cpython-311.pyc
      features.cpython-311.pyc
      labeling.cpython-311.pyc
      reconciliation.cpython-311.pyc
      train_model.cpython-311.pyc
    models
      experiment_log.json
      match_model.joblib
  tests
    __init__.py
    test_reconciliation.py
    __pycache__
      __init__.cpython-311.pyc
      test_reconciliation.cpython-311-pytest-9.1.1.pyc

## Why this isn’t just a light “looks okay” check
- Synthetic dataset injection is explicit in `data/generate_dataset.py`.
- The generator creates 200 students, 75 jobs, and 575 monetization events.
- Edge cases injected intentionally:
  - ~5% gateway/recorded amount mismatches.
  - 25 duplicate/partial payment events.
  - 10% missing skill scores in student profiles.
  - One zero-overlap job with `Cobol|Fortran|VHDL`.
- Detection tests prove the comparator fires:
  - `tests/test_reconciliation.py::test_mismatch_detected` verifies a known bad record (`A003`) is flagged.
  - `tests/test_reconciliation.py::test_charged_without_match_flagged` verifies charged-but-no-match cases are surfaced.
  - `tests/test_reconciliation.py::test_student_retains_application_on_failure` proves failed payments do not drop the application.
  - `tests/test_reconciliation.py::test_duplicates_detected` proves duplicate payment pairs are caught.

## Setup & rebuild-from-scratch instructions
1. Create and activate the virtual environment:
   ```powershell
   cd d:\Placemux-aiml\week3-task10
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
2. Regenerate data, train, evaluate, and build reports in order:
   ```powershell
   python run_pipeline.py
   ```
   - Order matters because `run_pipeline.py` generates `data/monetization_events.csv` first, then trains `src/models/match_model.joblib`, then evaluates, then writes `reports/evaluation_results.json`.
3. Start the API:
   ```powershell
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

## Demo walkthrough
1. Root / health check:
   ```powershell
   curl http://127.0.0.1:8000/
   curl http://127.0.0.1:8000/health
   ```
2. Happy-path explainability lookup:
   ```powershell
   curl http://127.0.0.1:8000/match/S010/J005
   ```
   Actual returned JSON includes:
   ```json
   {
     "student_id": "S010",
     "job_id": "J005",
     "ground_truth_label": 0,
     "baseline": {
       "overlap_count": 2,
       "total_required": 3,
       "overlap_ratio": 0.6667,
       "is_match": 1
     },
     "model": {
       "prediction": 0,
       "confidence": 0.937
     },
     "explanation": "[NOT A MATCH]. Matched on 2/3 required skills. Strong: AWS at level 5 (required 4) [OK]. Weak: Python at level 1 vs required 4. Missing: React. Model confidence: 0.94, driven mainly by weighted coverage score, min level delta, num missing required.",
     "skill_breakdown": {
       "matched": ["AWS at level 5 (required 4) [OK]"],
       "weak": ["Python at level 1 vs required 4"],
       "missing": ["React"]
     },
     "top_feature_drivers": [
       {"feature": "weighted_coverage_score", "importance": 0.4666},
       {"feature": "min_level_delta", "importance": 0.1902},
       {"feature": "num_missing_required", "importance": 0.1444},
       {"feature": "mean_level_delta", "importance": 0.0793},
       {"feature": "skill_overlap_count", "importance": 0.0403}
     ]
   }
   ```
3. Failure-handled path / reconciliation check:
   ```powershell
   curl http://127.0.0.1:8000/signoff/reconciliation
   ```
   This returns live `failed_applications`, `amount_mismatches`, `duplicate_events`, and `charged_without_match` counts.
4. Full evaluation summary:
   ```powershell
   curl http://127.0.0.1:8000/signoff/report
   ```

## Pitfalls checklist
- [x] No black box: `src/evaluation.py` and `src/reconciliation.py` contain the exact logic used.
- [x] Numbers not vibes: all metrics come from `reports/evaluation_results.json` and live `/signoff/report`.
- [x] Full population, not toy example: `reports/evaluation_results.json` uses 115 held-out test samples and `data/monetization_events.csv` includes 575 events.
- [x] No deferred failure-handling: `src/reconciliation.py::handle_payment_failure()` immediately maps failed payments to `refund_pending` or `payment_pending`.
- [x] Detector proven to fire: `tests/test_reconciliation.py::test_mismatch_detected` and `test_charged_without_match_flagged` exercise real bad records.
- [x] No single blended metric: precision, recall, and FPR are reported separately for baseline/model and pre-/post-monetization.

## Self-check answers
- Can you show it working live?
  Yes. Start the API with `uvicorn api.main:app --host 0.0.0.0 --port 8000` and use:
  - `/match/S010/J005`
  - `/signoff/report`
  - `/signoff/reconciliation`
- What happens if a payment fails halfway?
  This repo retains the student application and does not delete it. `src/reconciliation.py::handle_payment_failure()` sets `application_retained=True`; failed charges with `gateway_amount > 0` get `refund_initiated=True`.
- How do we know records match what the gateway says was collected?
  `src/reconciliation.py::validate_amounts()` compares `gateway_amount` to `recorded_amount` with tolerance and flags mismatches. The live `/signoff/reconciliation` endpoint returns those flags.
- Real-money or test mode?
  This is test mode on synthetic data. Before real money, the payment contract must be hardened, synthetic thresholds should be replaced with real gateway volume expectations, and the reconciliation logic should be wired to actual gateway event streams.

## Hand-off
Hand off this repo and the generated outputs to QA / monetization product engineering. The current hand-off package includes:
- `reports/evaluation_results.json`
- `reports/signoff_report.md`
- `src/reconciliation.py`
- `api/main.py`
- `tests/test_reconciliation.py`
- `data/monetization_events.csv`

Concrete suggestion: turn this one-off sign-off into a scheduled guardrail by rerunning `python run_pipeline.py` daily or on every gateway deployment and alerting if `/signoff/report` shows `post_precision` or `post_recall` regress from the previous baseline.

## Next steps
- Replace synthetic `data/monetization_events.csv` with the real gateway data contract once finalized.
- Add a threshold alert for `post_recall < 0.90` or `post_fpr` increase after real launch.
- Extend `/signoff/reconciliation` to record a daily audit log of mismatches and duplicate payment pairs.
- Verify the payment gateway contract end-to-end, not just the local reconciliation logic.
