"""
data_preprocessing.py
=====================
Generates realistic synthetic student-assessment data, cleans it, and
splits it into stratified train / validation / test sets.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_CSV = os.path.join(DATA_DIR, "synthetic_assessment_data.csv")

SUBJECTS = ["Mathematics", "Science", "English", "History", "Computer Science"]

# 1. Synthetic Data Generation

def generate_synthetic_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset of *n* student-assessment records.

    Relationships baked in
    ----------------------
    - Higher ``student_skill_level`` => higher scores & pass rates.
    - Higher ``difficulty_level`` (4-5) => lower pass rates.
    - Recent study (low ``days_since_last_study``) => better performance.
    - Gaussian noise prevents perfect separability (~78-82 % natural accuracy).

    Returns
    -------
    pd.DataFrame  –  saved to ``data/synthetic_assessment_data.csv``.
    """
    rng = np.random.RandomState(seed)

    # identifiers
    student_ids = rng.randint(1000, 9999, size=n)
    assessment_ids = rng.randint(100, 999, size=n)
    subjects = rng.choice(SUBJECTS, size=n)

    # core attributes
    difficulty_level = rng.randint(1, 6, size=n)                    # 1-5
    student_skill_level = rng.randint(1, 11, size=n)                # 1-10

    # Study behaviour (realistic ranges)
    study_sessions_count = rng.poisson(lam=5, size=n).clip(1, 20)
    time_spent_minutes = rng.normal(loc=60, scale=25, size=n).clip(10, 200)
    days_since_last_study = rng.exponential(scale=5, size=n).clip(0, 60).round(1)

    # Study hours per session (generate a list, then store mean + variance)
    study_hours_mean = np.zeros(n)
    study_hours_variance = np.zeros(n)
    for i in range(n):
        hours = rng.normal(loc=1.5 + student_skill_level[i] * 0.1,
                           scale=0.5,
                           size=study_sessions_count[i]).clip(0.25, 5)
        study_hours_mean[i] = hours.mean()
        study_hours_variance[i] = hours.var() if len(hours) > 1 else 0.0

    # Previous scores (generate a list, then store avg + trend)
    prev_scores_avg = np.zeros(n)
    prev_scores_trend = np.zeros(n)
    for i in range(n):
        num_prev = rng.randint(2, 8)
        base = 40 + student_skill_level[i] * 4 + rng.normal(0, 5, size=num_prev)
        # add an upward / downward drift
        drift = rng.normal(0, 1)
        scores = (base + np.arange(num_prev) * drift).clip(0, 100)
        prev_scores_avg[i] = scores.mean()
        if len(scores) > 1:
            x = np.arange(len(scores))
            prev_scores_trend[i] = np.polyfit(x, scores, 1)[0]  # slope
        else:
            prev_scores_trend[i] = 0.0

    # questions
    questions_attempted = rng.randint(10, 51, size=n)
    # Accuracy depends on skill & difficulty
    base_accuracy = (student_skill_level / 10 * 0.5 +
                     (6 - difficulty_level) / 5 * 0.3 +
                     0.1)
    base_accuracy += rng.normal(0, 0.08, size=n)
    base_accuracy = base_accuracy.clip(0.1, 0.98)
    questions_correct = (questions_attempted * base_accuracy).astype(int)
    questions_correct = np.minimum(questions_correct, questions_attempted)

    # assessment score
    # Score is a function of skill, difficulty, recency, effort, + noise
    skill_component = student_skill_level * 4.5                    # 4.5–45
    difficulty_penalty = difficulty_level * (-5)                    # -5 to -25
    recency_bonus = np.exp(-days_since_last_study / 10) * 15       # 0–15
    effort_bonus = (study_sessions_count * study_hours_mean) * 0.8 # variable
    accuracy_bonus = base_accuracy * 20                            # 2–19.6
    noise = rng.normal(0, 8, size=n)

    assessment_score = (30 + skill_component + difficulty_penalty +
                        recency_bonus + effort_bonus + accuracy_bonus + noise)
    assessment_score = assessment_score.clip(0, 100).round(1)

    # passed label
    passed = (assessment_score >= 60).astype(int)

    # assemble DataFrame
    df = pd.DataFrame({
        "student_id": student_ids,
        "assessment_id": assessment_ids,
        "subject": subjects,
        "difficulty_level": difficulty_level,
        "student_skill_level": student_skill_level,
        "time_spent_minutes": time_spent_minutes.round(1),
        "study_sessions_count": study_sessions_count,
        "study_hours_mean": study_hours_mean.round(3),
        "study_hours_variance": study_hours_variance.round(4),
        "days_since_last_study": days_since_last_study,
        "previous_scores_avg": prev_scores_avg.round(2),
        "previous_scores_trend": prev_scores_trend.round(4),
        "questions_attempted": questions_attempted,
        "questions_correct": questions_correct,
        "assessment_score": assessment_score,
        "passed": passed,
    })

    # Inject ~2 % missing values in a few columns for realism
    for col in ["time_spent_minutes", "study_hours_mean",
                "days_since_last_study", "previous_scores_avg"]:
        mask = rng.rand(n) < 0.02
        df.loc[mask, col] = np.nan

    # Inject a handful of duplicate rows
    dup_idx = rng.choice(n, size=int(n * 0.005), replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DEFAULT_CSV, index=False)
    print(f"[data_preprocessing] Saved {len(df)} rows -> {DEFAULT_CSV}")
    return df

# 2. Load Data

def load_data(filepath: str = DEFAULT_CSV) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame."""
    if not os.path.exists(filepath):
        print(f"[data_preprocessing] File not found: {filepath}")
        print("[data_preprocessing] Generating synthetic data …")
        return generate_synthetic_data()
    df = pd.read_csv(filepath)
    print(f"[data_preprocessing] Loaded {len(df)} rows from {filepath}")
    return df

# 3. Clean Data

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame:
      1. Drop exact duplicate rows.
      2. Handle missing values (median for numeric, mode for categorical).
      3. Clip outliers in ``time_spent_minutes`` using the IQR method.
      4. Encode ``subject`` with LabelEncoder.
    """
    df = df.copy()
    rows_before = len(df)

    df.drop_duplicates(inplace=True)
    print(f"[clean] Removed {rows_before - len(df)} duplicate rows.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            print(f"[clean] Filled {n_missing} NaNs in '{col}' with median={median_val:.2f}")

    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()[0]
            n_missing = df[col].isna().sum()
            df[col].fillna(mode_val, inplace=True)
            print(f"[clean] Filled {n_missing} NaNs in '{col}' with mode='{mode_val}'")

    # outlier clipping (IQR on time_spent_minutes)
    if "time_spent_minutes" in df.columns:
        Q1 = df["time_spent_minutes"].quantile(0.25)
        Q3 = df["time_spent_minutes"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        clipped = ((df["time_spent_minutes"] < lower) |
                   (df["time_spent_minutes"] > upper)).sum()
        df["time_spent_minutes"] = df["time_spent_minutes"].clip(lower, upper)
        print(f"[clean] Clipped {clipped} outliers in 'time_spent_minutes' "
              f"to [{lower:.1f}, {upper:.1f}]")

    # encode categorical (subject)
    if "subject" in df.columns:
        le = LabelEncoder()
        df["subject"] = le.fit_transform(df["subject"])
        print(f"[clean] Encoded 'subject': {dict(zip(le.classes_, le.transform(le.classes_)))}")

    print(f"[clean] Final shape: {df.shape}")
    return df

# 4. Split Data

def split_data(df: pd.DataFrame,
               target: str = "passed",
               test_size: float = 0.2,
               val_size: float = 0.1,
               random_state: int = 42):
    """
    Stratified train / validation / test split.

    Parameters
    ----------
    df : pd.DataFrame
    target : column name for the label
    test_size : fraction of total data reserved for testing
    val_size  : fraction of total data reserved for validation

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    feature_cols = [c for c in df.columns
                    if c not in [target, "student_id", "assessment_id"]]
    X = df[feature_cols]
    y = df[target]

    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: train vs val  (val_size is relative to the whole dataset)
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val, stratify=y_temp,
        random_state=random_state
    )

    print(f"[split] Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  "
          f"Test: {X_test.shape[0]}")
    print(f"[split] Target distribution -> "
          f"Train: {y_train.mean():.2%} pass | "
          f"Val: {y_val.mean():.2%} pass | "
          f"Test: {y_test.mean():.2%} pass")

    return X_train, X_val, X_test, y_train, y_val, y_test

# Main
if __name__ == "__main__":
    print("=" * 60)
    print("  Data Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: generate (or load) data
    if os.path.exists(DEFAULT_CSV):
        df = load_data(DEFAULT_CSV)
    else:
        df = generate_synthetic_data()

    print(f"\nRaw data shape : {df.shape}")
    print(f"Pass rate      : {df['passed'].mean():.2%}")
    print(f"Missing values :\n{df.isna().sum()[df.isna().sum() > 0]}\n")

    # Step 2: clean
    df_clean = clean_data(df)

    # Step 3: split
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

    print("\n[OK] Preprocessing complete.")
