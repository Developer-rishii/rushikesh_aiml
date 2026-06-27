import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix

class BaselineRanker:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self._is_ready = True
        
    def check_readiness(self):
        if not self._is_ready:
            raise RuntimeError("Baseline ranker is not ready or is stale.")
        return True

    def predict(self, df):
        """
        Dumb baseline: rank by raw overlap ratio.
        If overlap ratio >= threshold, predict 1 (good match), else 0.
        """
        self.check_readiness()
        # Ensure overlap_ratio exists
        if 'overlap_ratio' not in df.columns:
            raise ValueError("Dataframe must contain 'overlap_ratio' column.")
            
        predictions = (df['overlap_ratio'] >= self.threshold).astype(int)
        return predictions

def evaluate_baseline(df, ranker, split_name="Test"):
    ranker.check_readiness()
    y_true = df['is_good_match']
    y_pred = ranker.predict(df)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'fpr': fpr
    }
    
    print(f"[{split_name} Baseline] Precision: {precision:.3f}, Recall: {recall:.3f}, FPR: {fpr:.3f}")
    return metrics
