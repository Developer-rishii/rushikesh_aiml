# PlaceMux: Proctoring False-Positive Reduction Module

This repository contains the Machine Learning module for reducing false positives in PlaceMux's proctored skill tests. 

## Final Results

- **Baseline FPR**: ~5%
- **Model FPR**: ~0.9% 
- **False-Positive Reduction Achieved**: ~80% reduction in false positives while maintaining competitive recall.

## How to Run

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn xgboost shap fastapi uvicorn matplotlib pydantic
   ```
2. **Generate Data**:
   ```bash
   python data/generate_synthetic_sessions.py
   ```
3. **Run Baseline**:
   ```bash
   python src/baseline.py
   ```
4. **Train Model**:
   ```bash
   python src/train_model.py
   ```
5. **Evaluate Model**:
   ```bash
   python src/evaluate.py
   ```
6. **Start API**:
   ```bash
   uvicorn src.api:app --reload
   ```

## Live Walkthrough
Please refer to `demo/demo_script.md` for the step-by-step live demonstration, including edge-case handling and plain-English explainability.

## Dependency Mitigation Plan

**Current Status**: Real flagged-session data is not yet available.
**Mitigation Strategy**: 
1. **Development**: A realistic synthetic dataset was generated matching the expected 70% FP class imbalance. The loader `data/generate_synthetic_sessions.load_data` is configured to swap to a real CSV seamlessly.
2. **Production/Serving**: The FastAPI service (`src/api.py`) is designed to handle missing values and malformed schemas (returning 400 Bad Request rather than 500 crashes). If the inference engine fails or data is missing, the system gracefully falls back to the static `baseline_rule` implemented in `src/baseline.py`.
3. **Escalation**: If the upstream feed is delayed for >1 hour in production, an automated alert will page the Data Engineering team.
