"""
Training pipeline for LightGBM Ranker.
Saves training metrics to results/training_metrics.json for evidence.
All paths resolved relative to this file's location, not CWD.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import mlflow
import mlflow.lightgbm
import os
import json

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def eval_ndcg(y_true, y_pred, groups, k=10):
    """Evaluate mean nDCG@k across groups."""
    ndcg_scores = []
    idx = 0
    for g in groups:
        true_g = y_true[idx:idx+g]
        pred_g = y_pred[idx:idx+g]
        if len(true_g) > 1 and np.sum(true_g) > 0:
            ndcg_scores.append(ndcg_score([true_g], [pred_g], k=k))
        idx += g
    return np.mean(ndcg_scores) if ndcg_scores else 0.0

def train():
    root = get_project_root()
    print("Loading data...")
    df = pd.read_csv(os.path.join(root, 'data', 'interactions.csv'))

    df = df.sort_values(by=['candidate_id', 'job_id'])

    features = ['candidate_exp', 'candidate_skills', 'required_exp', 'required_skills', 'job_popularity']
    target = 'relevance'

    unique_candidates = np.array(df['candidate_id'].unique())
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

    mlflow_db = os.path.join(root, 'training', 'mlruns.db')
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment("Job_Recommendation_Ranking")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
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

        # Save model locally and model metadata
        models_dir = os.path.join(root, 'models')
        os.makedirs(models_dir, exist_ok=True)
        model.save_model(os.path.join(models_dir, 'lgbm_ranker.txt'))

        metadata = {
            "model_version": "1.0.0",
            "run_id": run_id,
            "registered_model_name": "LGBMRanker"
        }
        with open(os.path.join(models_dir, 'model_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Model saved and metadata written (run_id: {run_id}).")

        # Save training metrics to results/ for evidence
        results_dir = os.path.join(root, 'results')
        os.makedirs(results_dir, exist_ok=True)
        metrics = {
            "train_size": len(X_train),
            "test_size": len(X_test),
            "baseline_ndcg_10": round(baseline_ndcg, 4),
            "model_ndcg_10": round(model_ndcg, 4),
            "improvement": round(model_ndcg - baseline_ndcg, 4),
            "params": params
        }
        metrics_path = os.path.join(results_dir, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Training metrics saved to {metrics_path}")

if __name__ == "__main__":
    train()
