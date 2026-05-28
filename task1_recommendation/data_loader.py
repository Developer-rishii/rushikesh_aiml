import pandas as pd
import numpy as np
import random
import os
from typing import Dict, Any

# Constants
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'synthetic_students.csv')
NUM_STUDENTS = 1000
NUM_LEVELS = 50

def generate_synthetic_data(filepath: str) -> None:
    """
    Generates a realistic synthetic CSV with student level attempts.
    
    Args:
        filepath (str): The path to save the generated CSV.
    """
    np.random.seed(42)
    random.seed(42)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    records = []
    # Generate somewhat realistic data
    for student_id in range(1, NUM_STUDENTS + 1):
        num_attempts = random.randint(5, 25)
        # Levels they attempt
        levels_attempted = random.sample(range(1, NUM_LEVELS + 1), num_attempts)
        
        student_skill = random.uniform(40, 95)
        
        # Start time for student
        base_time = pd.Timestamp('2023-01-01') + pd.Timedelta(days=random.randint(0, 100))
        
        for i, level_id in enumerate(levels_attempted):
            score = min(100, max(0, int(random.gauss(student_skill, 15))))
            time_spent = max(1, int(random.gauss(30 - (student_skill/5), 10)))
            passed = bool(score >= 70)
            
            timestamp = base_time + pd.Timedelta(days=i)
            
            records.append({
                'student_id': f'STU_{student_id:04d}',
                'level_id': f'LVL_{level_id:03d}',
                'score': score,
                'time_spent_minutes': time_spent,
                'passed': passed,
                'timestamp': timestamp
            })
            
    df = pd.DataFrame(records)
    df.to_csv(filepath, index=False)

def load_data(filepath: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """
    Loads student data from a CSV. Generates synthetic data if it doesn't exist.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Raw dataframe containing student level attempts.
    """
    if not os.path.exists(filepath):
        print(f"File {filepath} not found. Generating synthetic data...")
        generate_synthetic_data(filepath)
        
    return pd.read_csv(filepath, parse_dates=['timestamp'])

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame. Keeps the highest score if a level was attempted multiple times.
    
    Args:
        df (pd.DataFrame): Raw DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.drop_duplicates()
    df = df.sort_values(by=['student_id', 'level_id', 'score'], ascending=[True, True, False])
    df = df.drop_duplicates(subset=['student_id', 'level_id'], keep='first')
    return df.reset_index(drop=True)

def build_student_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a pivot matrix (rows=students, cols=levels, values=score).
    Unattempted levels have a score of 0.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        
    Returns:
        pd.DataFrame: Pivot matrix.
    """
    matrix = df.pivot(index='student_id', columns='level_id', values='score')
    matrix = matrix.fillna(0)
    return matrix

def get_student_history(df: pd.DataFrame, student_id: str) -> Dict[str, float]:
    """
    Retrieves the history of completed levels and scores for a specific student.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame.
        student_id (str): The ID of the student.
        
    Returns:
        Dict[str, float]: Dictionary mapping level_id to score.
    """
    student_df = df[df['student_id'] == student_id]
    history = dict(zip(student_df['level_id'], student_df['score']))
    return history
