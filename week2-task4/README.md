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
