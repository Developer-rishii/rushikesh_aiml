# PlaceMux Task 5 – Marketplace Integration & Company Portal v1

This module contains the baseline matching engine and explainable Machine Learning pipeline for PlaceMux Task 5. 

## Project Structure
- `data/`: Contains `jobs.csv` and `students.csv` generated realistically.
- `src/`: 
  - `data_generation.py`: Script to generate sample jobs and students data.
  - `baseline_matcher.py`: Core logic for match vector generation, scoring, explainability, and the secondary eligibility gate (`minimum_skill_score`).
  - `ranker.py`: Engine to filter eligible candidates and sort them by score, using `average_verified_skill_score` and experience gap as tiebreakers.
  - `ml_model.py`: Feature engineering, new noisy label generation, and Logistic Regression training pipeline.
  - `edge_cases.py`: Script validating 7 explicit edge cases.
  - `generate_evidence.py`: Script generating statistics and plotting match distributions.
  - `log_experiment.py`: Utility to automatically log experiment runs to `experiments/experiment_log.csv`.
  - `api.py`: FastAPI service providing endpoints for matching and ranking.
- `docs/demo_evidence/`: Charts and JSON exports proving correct behavior with real data.
- `notebooks/Task5_Workflow.ipynb`: A step-by-step Jupyter Notebook to explore the matching behavior interactively.
- `models/`: Pickled ML models and evaluation metrics.

## Running the End-to-End Demo

### 1. Generating Data
The datasets have already been generated in the `data/` folder. If you wish to regenerate:
```bash
python src/data_generation.py
```

### 2. Testing Edge Cases
To prove the system gracefully handles zero skills, missing data, boundary cases, and minimum skill score gates:
```bash
python src/edge_cases.py
```

### 3. Training the ML Model
To generate features and train the Logistic Regression model, resulting in metric outputs:
```bash
python src/ml_model.py
```

### 4. Exploring the Demo Evidence
Run the script to generate dataset stats, rank examples, and distribution charts:
```bash
python src/generate_evidence.py
```
Check `docs/demo_evidence/` for visual charts showing the distribution of match scores and pass/fail thresholds.

### 5. Running the API Service
Start the FastAPI application:
```bash
uvicorn src.api:app --reload
```
You can access the interactive swagger docs at `http://localhost:8000/docs` to test:
- `POST /match`: Input a `job_id` and `student_id`. View the `match_vector`, plain English `reason`, threshold validation, and prediction.
- `POST /rank-candidates`: Input a `job_id` to get a descending sorted list of eligible candidates.

### 6. Interactive Jupyter Notebook
Launch Jupyter to explore the notebook:
```bash
jupyter notebook notebooks/Task5_Workflow.ipynb
```
This notebook provides cell-by-cell execution showing the progression from loading data to ranking candidates.

---

## Instructor Feedback Compliance Checklist
- [x] Follow the study guide exactly.
- [x] Prioritize explainability over complexity (Rule-based Baseline + Logistic Regression).
- [x] Build a baseline first before any ML model.
- [x] Every decision must be measurable with metrics.
- [x] Every match must have a plain-English explanation.
- [x] Use real-shaped sample data (50+ jobs, 300+ students).
- [x] Ensure the final system is demoable end-to-end.
- [x] Handled all specified Edge Cases.
- [x] Provided real data quality evidence via charts and reports.
