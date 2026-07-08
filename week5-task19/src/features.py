import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr

def clean_data(responses_df: pd.DataFrame) -> pd.DataFrame:
    df = responses_df.copy()
    # Drop malformed rows
    df = df.dropna(subset=['correct', 'item_id', 'student_id'])
    # Fix invalid types/values
    df['correct'] = pd.to_numeric(df['correct'], errors='coerce')
    df = df.dropna(subset=['correct'])
    df['correct'] = df['correct'].astype(int)
    
    df['time_spent_sec'] = pd.to_numeric(df['time_spent_sec'], errors='coerce')
    df = df.dropna(subset=['time_spent_sec'])
    df = df[df['time_spent_sec'] > 0]
    
    return df

def extract_features(items_df: pd.DataFrame, responses_df: pd.DataFrame) -> pd.DataFrame:
    resp_df = clean_data(responses_df)
    
    # Calculate student total scores
    student_scores = resp_df.groupby('student_id')['correct'].sum().reset_index()
    student_scores.columns = ['student_id', 'total_score']
    
    resp_merged = resp_df.merge(student_scores, on='student_id', how='left')
    
    features_list = []
    for item_id, group in resp_merged.groupby('item_id'):
        response_count = len(group)
        if response_count == 0:
            continue
            
        p_value = group['correct'].mean()
        time_variance = group['time_spent_sec'].var(ddof=0) if response_count > 1 else 0.0
        score_variance = group['correct'].var(ddof=0) if response_count > 1 else 0.0
        
        # Point-biserial correlation
        rest_of_test = group['total_score'] - group['correct']
        
        if score_variance == 0 or rest_of_test.var(ddof=0) == 0:
            pb_corr = 0.0
        else:
            try:
                pb_corr, _ = pointbiserialr(group['correct'], rest_of_test)
                if np.isnan(pb_corr):
                    pb_corr = 0.0
            except:
                pb_corr = 0.0
                
        if response_count >= 4:
            q25 = rest_of_test.quantile(0.25)
            q75 = rest_of_test.quantile(0.75)
            # handle cases where quantiles are equal by strictly taking top and bottom
            # or just use the threshold
            bottom_25_mask = rest_of_test <= q25
            top_25_mask = rest_of_test >= q75
            bottom_25_correct_rate = group[bottom_25_mask]['correct'].mean() if bottom_25_mask.sum() > 0 else 0.0
            top_25_correct_rate = group[top_25_mask]['correct'].mean() if top_25_mask.sum() > 0 else 0.0
        else:
            bottom_25_correct_rate = 0.0
            top_25_correct_rate = 0.0
            
        features_list.append({
            'item_id': item_id,
            'response_count': response_count,
            'p_value': p_value,
            'time_variance': time_variance,
            'score_variance': score_variance,
            'point_biserial_corr': pb_corr,
            'bottom_25_correct_rate': bottom_25_correct_rate,
            'top_25_correct_rate': top_25_correct_rate
        })
        
    if not features_list:
        features_df = pd.DataFrame(columns=[
            'item_id', 'response_count', 'p_value', 'time_variance', 
            'score_variance', 'point_biserial_corr', 'bottom_25_correct_rate', 'top_25_correct_rate'
        ])
    else:
        features_df = pd.DataFrame(features_list)
    
    # Merge with items
    # Only keep necessary columns from items to avoid label leakage into features dataframe accidentally
    # Actually we need split, is_weak_item for modeling, and subject as a feature.
    meta_cols = ['item_id', 'subject', 'is_weak_item', 'split']
    items_unique = items_df[meta_cols].drop_duplicates(subset=['item_id'])
    
    final_df = items_unique.merge(features_df, on='item_id', how='left')
    final_df['response_count'] = final_df['response_count'].fillna(0)
    
    # Fill NaNs for items with 0 responses (will be handled by "needs_more_data" logic downstream)
    final_df = final_df.fillna(0)
    
    return final_df
