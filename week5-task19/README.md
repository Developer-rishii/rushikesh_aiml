# PlaceMux Item-Bank Quality Support (Week 5 Task 19)

This module provides ML-based detection of weak or unreliable assessment items (questions) using principles from Classical Test Theory and Item Response Theory.

## Performance Metrics

*These numbers are auto-generated from `reports/metrics.json` during the pipeline run.*

| Metric | Dumb Baseline | ML Model |
|---|---|---|
| **Precision** | 0.3333 | 0.8667 |
| **Recall** | 0.1176 | 0.7647 |
| **FPR** | 0.0784 | 0.0392 |
| **AUC** | N/A | 0.9746 |

## Quick Start

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the end-to-end pipeline (generates synthetic data, trains model, scores items):
   ```bash
   python run_pipeline.py
   ```
3. Run tests:
   ```bash
   python -m pytest tests/ -v
   ```
4. Start the API:
   ```bash
   uvicorn src.api:app --reload
   ```

## API Endpoints

- `GET /items/{item_id}/quality` - Get item quality with plain-English explanation.
- `GET /admin/weak-items` - View paginated list of all weak items. Filter by `?subject=` or `?college_id=`.
- `GET /college/{college_id}/weak-items` - View weak items specific to a college. Enforces data isolation.
- `GET /report` - View full JSON metrics for the latest pipeline run.

## Extending this module (For the Next Team)
- **Adding features**: Add new extraction logic in `src/features.py`. The pipeline automatically one-hot encodes `subject` and fits on the rest.
- **Model Tuning**: Update the `param_grid` in `src/model.py`.
- **Explainability**: Adjust the threshold and rules in `src/explainability.py` for new features.
