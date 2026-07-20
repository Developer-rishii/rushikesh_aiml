import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import mlflow
import mlflow.lightgbm
import os

def eval_ndcg(y_true, y_pred, groups, k=10):
    """Evaluate mean nDCG@k across groups."""
    ndcg_scores = []
    idx = 0
    for g in groups:
        true_g = y_true[idx:idx+g]
        pred_g = y_pred[idx:idx+g]
        # Only evaluate if more than 1 item and at least one relevant item
        if len(true_g) > 1 and np.sum(true_g) > 0:
            ndcg_scores.append(ndcg_score([true_g], [pred_g], k=k))
        idx += g
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def train():
    print("Loading data...")
    df = pd.read_csv('data/interactions.csv')

    # Sort by candidate_id for LightGBM grouping
    df = df.sort_values(by=['candidate_id', 'job_id'])

    features = ['candidate_exp', 'candidate_skills', 'required_exp', 'required_skills', 'job_popularity']
    target = 'relevance'

    # Simple train-test split (in reality, temporal split is better)
    # To keep groups intact, we split by candidate_id
    unique_candidates = df['candidate_id'].unique()
    train_cands, test_cands = train_test_split(unique_candidates, test_size=0.2, random_state=42)

    train_df = df[df['candidate_id'].isin(train_cands)].sort_values(by='candidate_id')
    test_df = df[df['candidate_id'].isin(test_cands)].sort_values(by='candidate_id')

    X_train = train_df[features]
    y_train = train_df[target]
    group_train = train_df.groupby('candidate_id').size().values

    X_test = test_df[features]
    y_test = test_df[target]
    group_test = test_df.groupby('candidate_id').size().values

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Job_Recommendation_Ranking")

    with mlflow.start_run():
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [10],
            'learning_rate': 0.1,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'random_state': 42
        }

        train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
        valid_data = lgb.Dataset(X_test, label=y_test, group=group_test, reference=train_data)

        print("Training LightGBM Ranker...")
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[train_data, valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )

        # Baseline: Recommend by purely job popularity
        baseline_preds = X_test['job_popularity'].values
        baseline_ndcg = eval_ndcg(y_test.values, baseline_preds, group_test, k=10)

        # Model Predictions
        test_preds = model.predict(X_test)
        model_ndcg = eval_ndcg(y_test.values, test_preds, group_test, k=10)

        print(f"Baseline (Popularity) nDCG@10: {baseline_ndcg:.4f}")
        print(f"LightGBM nDCG@10: {model_ndcg:.4f}")

        # Logging
        mlflow.log_params(params)
        mlflow.log_metric("baseline_ndcg_10", baseline_ndcg)
        mlflow.log_metric("model_ndcg_10", model_ndcg)

        # Log Model
        mlflow.lightgbm.log_model(model, "model", registered_model_name="LGBMRanker")
        
        # Save model locally for fast API startup
        os.makedirs('models', exist_ok=True)
        model.save_model('models/lgbm_ranker.txt')
        print("Model saved and logged to MLflow.")

if __name__ == "__main__":
    train()
