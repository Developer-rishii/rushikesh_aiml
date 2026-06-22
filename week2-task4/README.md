# PlaceMux Explainability Module

This module implements an explainable candidate-job matching system for PlaceMux. It provides both a rule-based baseline model and an explainable ML model (Logistic Regression). An explainability engine produces human-readable reasons for why a candidate matches a job.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate data:
   ```bash
   python src/preprocessing/generate_data.py
   ```

3. Train the model:
   ```bash
   python src/models/train.py
   ```

4. Run the API server:
   ```bash
   uvicorn app:app --reload
   ```

5. Run the end-to-end Demo:
   ```bash
   python demo.py
   ```

6. Run Tests:
   ```bash
   pytest tests/
   ```

## Approach

The task to construct an Explainability Module for the candidate-job matching system was solved using the following structured approach:

1. **Synthetic Data Generation & Preparation**
   Since the ranking system serves as an upstream dependency, we developed a `generate_data.py` script to generate 3000+ realistic synthetic pairs of candidates and jobs. This gave us organic distributions for skills, experience, verified scores, and minimum requirements to use for training.

2. **Feature Engineering**
   Raw lists of skills are not directly consumable by transparent linear models. We engineered features to quantify the candidate-to-job fit:
   - `skill_overlap_percentage`
   - `number_of_matching_skills`
   - `number_of_missing_skills`
   - `verified_score` and `experience_years`

3. **Model Development & Baselines**
   - **Baseline Rule-Based Matcher:** We first implemented the required baseline, scoring matches exactly by the formula: `(matching_skills / required_skills) * 100`.
   - **Explainable ML (Logistic Regression):** To go beyond a simple ratio, we trained a Logistic Regression model on the engineered features. By explicitly avoiding black-box methods like Neural Networks and LLMs, we guarantee 100% deterministic, linear explainability.

4. **Explainability Engine (The Core)**
   Instead of using generative AI which can hallucinate, we mapped the deterministic output of the feature engineering step into carefully crafted plain-English text templates. The engine seamlessly identifies exactly *which* skills matched, *which* are missing, and *why* the qualification score threshold was or wasn't met.

5. **Rigorous Evaluation & Experiment Tracking**
   We evaluate the models with standard classification metrics (Precision, Recall, False Positive Rate, F1, Accuracy) and automatically persist runs to `experiments/experiment_log.csv` ensuring full reproducibility.

6. **Production-Ready API**
   The entire system was wrapped inside a clean `FastAPI` layer simulating the real PlaceMux microservice architecture, allowing upstream services to query `/api/predict` and immediately receive the human-readable explanation payload.
