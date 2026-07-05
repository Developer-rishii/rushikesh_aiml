"""
baseline.py

Baseline rule-based ranker (uses only skill overlap).
"""

import numpy as np

class BaselineRanker:
    """A dumb ranker that scores jobs solely based on skill_overlap_ratio."""
    
    def predict_proba(self, features_df):
        """Returns dummy probabilities proportional to skill overlap."""
        # Ensure we return a 2D array [P(False), P(True)] for sklearn compatibility
        p_true = features_df["skill_overlap_ratio"].values
        # Add tiny noise to break ties deterministically
        np.random.seed(42)
        p_true = np.clip(p_true + np.random.normal(0, 0.01, size=len(p_true)), 0, 1)
        
        p_false = 1.0 - p_true
        return np.vstack((p_false, p_true)).T

if __name__ == "__main__":
    from sklearn.metrics import precision_score, recall_score, confusion_matrix
    from features import FeatureEngineer
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit
    
    # Quick standalone test of baseline
    print("Testing Baseline Ranker...")
    fe = FeatureEngineer()
    fe.load_context()
    
    outcomes = pd.read_csv("data/outcomes.csv")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(outcomes, groups=outcomes["student_id"]))
    
    train_outcomes = outcomes.iloc[train_idx]
    test_outcomes = outcomes.iloc[test_idx]
    
    fe.fit(train_outcomes)
    X_test = fe.transform(test_outcomes)
    y_test = test_outcomes["was_hired"].values
    
    ranker = BaselineRanker()
    y_pred_proba = ranker.predict_proba(X_test)[:, 1]
    
    # Threshold at 0.5
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    print(f"Baseline Precision: {prec:.3f}")
    print(f"Baseline Recall:    {rec:.3f}")
    print(f"Baseline FPR:       {fpr:.3f}")
