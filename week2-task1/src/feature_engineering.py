import pandas as pd
from typing import Dict, List, Tuple

def load_data(students_path: str, jobs_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads student and job datasets from CSV."""
    students_df = pd.read_csv(students_path)
    jobs_df = pd.read_csv(jobs_path)
    return students_df, jobs_df

def preprocess_jobs(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the jobs dataframe.
    - Converts 'Required Skills' from comma-separated string to list.
    - Converts 'Minimum Skill Scores' from comma-separated string to list of integers.
    - Creates a dictionary mapping each required skill to its minimum score.
    """
    jobs_processed = jobs_df.copy()
    
    # Fill NaN with empty string/zero where appropriate just in case
    jobs_processed['Required Skills'] = jobs_processed['Required Skills'].fillna('')
    jobs_processed['Minimum Skill Scores'] = jobs_processed['Minimum Skill Scores'].fillna('')
    
    def parse_skills_and_scores(row) -> Dict[str, int]:
        skills = [s.strip() for s in row['Required Skills'].split(',') if s.strip()]
        scores_str = [s.strip() for s in str(row['Minimum Skill Scores']).split(',') if s.strip()]
        
        # Ensure we have scores for all skills, default to 0 if missing
        scores = []
        for score in scores_str:
            try:
                scores.append(int(score))
            except ValueError:
                scores.append(0)
                
        # Pad scores if there are fewer scores than skills
        while len(scores) < len(skills):
            scores.append(0)
            
        return dict(zip(skills, scores))
        
    jobs_processed['Skill Requirements'] = jobs_processed.apply(parse_skills_and_scores, axis=1)
    
    return jobs_processed

def preprocess_students(students_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the students dataframe.
    - Creates a dictionary mapping skills to their verified scores for easy lookup.
    """
    students_processed = students_df.copy()
    
    def extract_skills(row) -> Dict[str, int]:
        skills = {}
        for col in row.index:
            if str(col).startswith('Verified ') and str(col).endswith(' Score'):
                skill_name = str(col).replace('Verified ', '').replace(' Score', '')
                skills[skill_name] = int(row[col])
        return skills
        
    students_processed['Skills'] = students_processed.apply(extract_skills, axis=1)
    return students_processed

def get_feature_spaces(students_path: str, jobs_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end feature extraction for both datasets."""
    students_df, jobs_df = load_data(students_path, jobs_path)
    students_features = preprocess_students(students_df)
    jobs_features = preprocess_jobs(jobs_df)
    return students_features, jobs_features
