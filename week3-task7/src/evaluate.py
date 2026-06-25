import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from matcher import JobMatcher
from baseline import get_baseline_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')

def load_data(split='test'):
    pairs = pd.read_csv(os.path.join(DATA_DIR, f'{split}_pairs.csv'))
    candidates = pd.read_csv(os.path.join(DATA_DIR, 'candidates.csv'))
    jobs = pd.read_csv(os.path.join(DATA_DIR, 'jobs.csv'))
    
    # Merge
    data = pairs.merge(candidates, on='Candidate ID').merge(jobs, on='Job ID')
    return data

def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return acc, prec, rec, f1, fpr

def main():
    print("Loading test data...")
    test_df = load_data('test')
    
    y_true = test_df['is_match'].values
    
    matcher = JobMatcher(model_path='../models/logistic_regression.joblib')
    
    y_pred_baseline = []
    y_pred_tuned = []
    
    print("Generating predictions...")
    for _, row in test_df.iterrows():
        cand = {
            'Candidate ID': row['Candidate ID'],
            'Skills': row['Skills'],
            'Experience Years': row['Experience Years'],
            'Education': row['Education'],
            'Certifications': row['Certifications'],
            'Projects': row['Projects']
        }
        job = {
            'Job ID': row['Job ID'],
            'Required Skills': row['Required Skills'],
            'Preferred Skills': row['Preferred Skills'],
            'Experience Requirement': row['Experience Requirement'],
            'Education Requirement': row['Education Requirement']
        }
        
        # Baseline prediction
        b_score = get_baseline_score(cand, job)
        y_pred_baseline.append(1 if b_score >= 0.5 else 0)
        
        # Tuned prediction
        match_res = matcher.match(cand, job)
        y_pred_tuned.append(1 if match_res['final_score'] >= 50.0 else 0)
        
    print("Evaluating models...")
    b_acc, b_prec, b_rec, b_f1, b_fpr = evaluate_predictions(y_true, y_pred_baseline)
    t_acc, t_prec, t_rec, t_f1, t_fpr = evaluate_predictions(y_true, y_pred_tuned)
    
    print("\n================ EVALUATION REPORT ================\n")
    print(f"{'Metric':<15} | {'Baseline':<10} | {'Tuned':<10}")
    print("-" * 43)
    print(f"{'Accuracy':<15} | {b_acc:<10.4f} | {t_acc:<10.4f}")
    print(f"{'Precision':<15} | {b_prec:<10.4f} | {t_prec:<10.4f}")
    print(f"{'Recall':<15} | {b_rec:<10.4f} | {t_rec:<10.4f}")
    print(f"{'F1 Score':<15} | {b_f1:<10.4f} | {t_f1:<10.4f}")
    print(f"{'FPR':<15} | {b_fpr:<10.4f} | {t_fpr:<10.4f}")
    print("-" * 43)
    
    # Save the report
    report_path = os.path.join(BASE_DIR, '../experiments/evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("Metric | Baseline | Tuned\n")
        f.write(f"Accuracy | {b_acc:.4f} | {t_acc:.4f}\n")
        f.write(f"Precision | {b_prec:.4f} | {t_prec:.4f}\n")
        f.write(f"Recall | {b_rec:.4f} | {t_rec:.4f}\n")
        f.write(f"F1 | {b_f1:.4f} | {t_f1:.4f}\n")
        f.write(f"FPR | {b_fpr:.4f} | {t_fpr:.4f}\n")
    print(f"\nSaved evaluation report to {report_path}")
    
if __name__ == '__main__':
    main()
