import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, f1_score

def evaluate_and_log(run_id, model_name, y_true, y_pred):
    """
    Evaluate the model and log results to experiments/experiment_log.csv
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # False Positive Rate
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        fpr = 0.0 # fallback

    log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'experiments')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'experiment_log.csv')
    
    # Create or append
    file_exists = os.path.isfile(log_file)
    
    record = {
        "run_id": run_id,
        "model": model_name,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "fpr": round(fpr, 2),
        "f1": round(f1, 2),
        "accuracy": round(accuracy, 2)
    }
    
    df_record = pd.DataFrame([record])
    
    if file_exists:
        df_record.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df_record.to_csv(log_file, mode='w', header=True, index=False)
        
    print(f"Logged experiment {run_id} for {model_name}.")
    return record
