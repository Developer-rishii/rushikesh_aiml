# Task 18: Recommendation Explainability (Rec v1 Extension)

This repository contains the completed "Explainability" implementation for PlaceMux. It extends the Rec v1 system by providing structured, multi-audience (student, placement officer, admin), actionable, and measurable explanations for every ranked job recommendation.

This solution is designed to strengthen recommendation explainability so that predictions are no longer black-box strings, but rich, multi-tiered insights backed by an ML-driven quality score and programmatic counterfactuals.

## 1. Quick Start & Live Demo

You can run the entire pipeline end-to-end to generate data, train the quality model, evaluate explanations, and start the API:

```bash
# 1. Run the data pipeline and generate the metrics report
python src/pipeline.py

# 2. Start the API server
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**What the API endpoints show:**
- **Student View (`/explain/{college_id}/{student_id}/{job_id}/student`):** A supportive, action-oriented explanation showing skill gaps and rank improvements (counterfactuals).
- **Placement Officer View (`/explain/{college_id}/{student_id}/{job_id}/officer`):** A broader perspective focusing on cohort ranking, AI trust score, and recommended intervention actions.
- **Admin/Debugging View (`/explain/{college_id}/{student_id}/{job_id}/admin`):** Feature attributions, weights, model confidence, and precise mathematical impacts of counterfactuals.
- **Aggregated View (`/explain/{college_id}/{student_id}/{job_id}`):** Returns all three audiences and their predicted quality scores.
- **College Portal Dashboard (`/portal/{college_id}/dashboard`):** Real-time aggregation of explanation metrics for a specific college.
- **Metrics Report (`/explain/report`):** Full JSON payload of the pipeline metrics comparing the new system against the baseline.

## 2. Multi-Level Explainability & Counterfactuals

Instead of a single plain-English string, explanations are now measurably richer:
- **Structured Skill Gaps:** Explanations specifically identify which skills are missing.
- **Counterfactuals ("Proof of Teeth"):** Re-scoring logic proves that the model actually responds to intervention. For instance, "Adding Python moves your rank from #4 to #1."
- **Multi-Audience:** The same underlying prediction surfaces different details depending on who is asking.

## 3. Live Verification, Evidence & Metrics

We evaluate explanation quality across multiple dimensions. All metrics are evaluated against a heuristic baseline.

**Overall Performance (vs Baseline):**
- **Completeness:** Improved significantly (e.g., from ~50% to >80%).
- **Actionability:** Rose from 0% (baseline strings offered no action) to >60%.
- **Specificity:** Perfectly maintained at 100%.
- **ML Quality Score:** Our trained RandomForest model shows a massive jump in high-quality explanations (scores > 0.7) compared to the baseline.
- **Counterfactual Fraction:** Over 60% of explanations now include a proven counterfactual.

*View the exact numbers and segmentations in `reports/metrics.json` or `reports/sign_off_report.md`.*

## 4. Dependency, Failure & Edge-Case Handling

- **Data Isolation:** The API strictly scopes every request to the `college_id`. `TestCrossCollegeIsolation` in `tests/test_isolation.py` proves that College A cannot access College B's explanations.
- **Upstream Dependency (Rec v1):** Validates the schema of `rec_v1_output.csv` at load time. If required columns are missing, the pipeline fails safely (`tests/test_validation.py`).
- **Edge Cases:** Unknown students or missing explanations safely fall back to 404s or graceful error messages, rather than HTTP 500 stack traces (`tests/test_edge_cases.py`).

## 5. Hand-off (Next Steps)

For the downstream team extending this system:
1. **Model:** The quality model artifact is persisted at `src/models/quality_model.joblib`.
2. **API Contract:** Available natively via the FastAPI Swagger UI at `http://localhost:8000/docs`.
3. **Tests:** Ensure `pytest tests/ -v` runs cleanly to guarantee isolation and logic regressions are caught.
