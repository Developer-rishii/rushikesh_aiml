import os
import pandas as pd
import joblib

from src.data_generation import generate_data
from src.features import extract_features
from src.model import train_and_evaluate

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    model_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    
    # 1. Generate Data
    generate_data(data_dir)
    
    # 2. Extract Features
    items_df = pd.read_csv(os.path.join(data_dir, 'items.csv'))
    responses_df = pd.read_csv(os.path.join(data_dir, 'responses.csv'))
    features_df = extract_features(items_df, responses_df)
    
    # 3. Train Model and Evaluate
    metrics = train_and_evaluate(features_df, model_dir, reports_dir)
    print("Metrics generated:", metrics)
    
    # 4. Score all items and save to features_scored.csv
    # Load model and columns
    model = joblib.load(os.path.join(model_dir, 'weak_item_model.pkl'))
    feature_cols = joblib.load(os.path.join(model_dir, 'feature_cols.pkl'))
    
    # Score items with >= 20 responses
    score_mask = features_df['response_count'] >= 20
    score_df = features_df[score_mask].copy()
    
    # Encode subjects
    df_encoded = pd.get_dummies(score_df, columns=['subject'])
    X_input = pd.DataFrame(0, index=df_encoded.index, columns=feature_cols)
    for col in df_encoded.columns:
        if col in feature_cols:
            X_input[col] = df_encoded[col]
            
    # Predict
    score_df['model_is_weak'] = model.predict(X_input)
    score_df['model_confidence'] = model.predict_proba(X_input)[:, 1]
    
    # Merge back to original features_df
    features_scored = features_df.merge(
        score_df[['item_id', 'model_is_weak', 'model_confidence']], 
        on='item_id', 
        how='left'
    )
    
    features_scored['model_is_weak'] = features_scored['model_is_weak'].fillna(False).astype(bool)
    features_scored['model_confidence'] = features_scored['model_confidence'].fillna(0.0)
    
    features_scored.to_csv(os.path.join(data_dir, 'features_scored.csv'), index=False)
    
    # 5. Generate README
    readme_content = f"""# PlaceMux Item-Bank Quality Support (Week 5 Task 19)

This module provides ML-based detection of weak or unreliable assessment items (questions) using principles from Classical Test Theory and Item Response Theory.

## Performance Metrics

*These numbers are auto-generated from `reports/metrics.json` during the pipeline run.*

| Metric | Dumb Baseline | ML Model |
|---|---|---|
| **Precision** | {metrics['baseline']['precision']} | {metrics['model']['precision']} |
| **Recall** | {metrics['baseline']['recall']} | {metrics['model']['recall']} |
| **FPR** | {metrics['baseline']['fpr']} | {metrics['model']['fpr']} |
| **AUC** | N/A | {metrics['model']['auc']} |

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

- `GET /items/{{item_id}}/quality` - Get item quality with plain-English explanation.
- `GET /admin/weak-items` - View paginated list of all weak items. Filter by `?subject=` or `?college_id=`.
- `GET /college/{{college_id}}/weak-items` - View weak items specific to a college. Enforces data isolation.
- `GET /report` - View full JSON metrics for the latest pipeline run.

## Extending this module (For the Next Team)
- **Adding features**: Add new extraction logic in `src/features.py`. The pipeline automatically one-hot encodes `subject` and fits on the rest.
- **Model Tuning**: Update the `param_grid` in `src/model.py`.
- **Explainability**: Adjust the threshold and rules in `src/explainability.py` for new features.
"""
    with open(os.path.join(base_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
        
    print("Pipeline complete. Data and README saved.")

if __name__ == "__main__":
    main()
