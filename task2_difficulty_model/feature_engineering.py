"""
feature_engineering.py
======================
Scikit-learn-style FeatureEngineer that creates exactly 13 predictive
features from raw assessment data **without data leakage**.

fit()  -> learns statistics from training data only.
transform() -> applies learned statistics to any split.
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Creates 13 engineered features for assessment-pass prediction.

    Core features  (5): student_ability_score, learning_consistency,
                        days_since_last_study, difficulty_tier, learning_velocity
    Interaction    (5): skill_difficulty_gap, time_efficiency,
                        preparation_intensity, recency_weighted_score,
                        difficulty_adjusted_ability
    Ratio features (3): accuracy_ratio, consistency_score, normalized_time
    """

    FEATURE_NAMES = [
        # Core
        "student_ability_score",
        "learning_consistency",
        "days_since_last_study",
        "difficulty_tier",
        "learning_velocity",
        # Interaction
        "skill_difficulty_gap",
        "time_efficiency",
        "preparation_intensity",
        "recency_weighted_score",
        "difficulty_adjusted_ability",
        # Ratio
        "accuracy_ratio",
        "consistency_score",
        "normalized_time",
    ]

    def __init__(self):
        self._fitted = False
        # Learned statistics (populated during fit)
        self._global_ability_mean: float = 0.0
        self._global_velocity_mean: float = 0.0
        self._global_consistency_mean: float = 0.0

    # Public API
    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """
        Learn aggregate statistics from **training data only**.

        These stats are used as fallbacks during transform() so that
        validation / test sets never leak information back into features.
        """
        self._global_ability_mean = df["previous_scores_avg"].mean()
        self._global_velocity_mean = df["previous_scores_trend"].mean()
        self._global_consistency_mean = df["study_hours_variance"].mean()
        self._fitted = True
        print(f"[FeatureEngineer] fit complete -- "
              f"ability mean={self._global_ability_mean:.2f}, "
              f"velocity mean={self._global_velocity_mean:.4f}, "
              f"consistency mean={self._global_consistency_mean:.4f}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build 13 features from *df* using statistics learned during fit().
        Returns a new DataFrame containing **only** the 13 engineered columns.
        """
        if not self._fitted:
            raise RuntimeError("FeatureEngineer.fit() must be called before transform().")

        out = pd.DataFrame(index=df.index)

        # Core features
        # 1. student_ability_score – rolling avg of previous assessment scores
        out["student_ability_score"] = df["previous_scores_avg"].fillna(
            self._global_ability_mean
        )

        # 2. learning_consistency – std dev of study hours (low = consistent)
        #    We stored variance; take sqrt for std dev.
        out["learning_consistency"] = np.sqrt(
            df["study_hours_variance"].fillna(self._global_consistency_mean)
        )

        # 3. days_since_last_study – clipped at 30
        out["days_since_last_study"] = df["days_since_last_study"].clip(upper=30)

        # 4. difficulty_tier – Easy(1-2)=0, Medium(3)=1, Hard(4-5)=2
        out["difficulty_tier"] = pd.cut(
            df["difficulty_level"],
            bins=[0, 2, 3, 5],
            labels=[0, 1, 2],
            include_lowest=True,
        ).astype(int)

        # 5. learning_velocity – slope of student's score history
        out["learning_velocity"] = df["previous_scores_trend"].fillna(
            self._global_velocity_mean
        )

        # Interaction features
        # 6. skill_difficulty_gap
        out["skill_difficulty_gap"] = (
            df["student_skill_level"] - df["difficulty_level"]
        )

        # 7. time_efficiency – score per minute
        time_safe = df["time_spent_minutes"].replace(0, np.nan).fillna(1)
        out["time_efficiency"] = df["assessment_score"] / time_safe

        # 8. preparation_intensity – total prep volume
        out["preparation_intensity"] = (
            df["study_sessions_count"] * df["study_hours_mean"].fillna(
                df["study_hours_mean"].median()
            )
        )

        # 9. recency_weighted_score – exponential decay
        days = df["days_since_last_study"].fillna(30)
        out["recency_weighted_score"] = (
            out["student_ability_score"] * np.exp(-days / 7)
        )

        # 10. difficulty_adjusted_ability
        out["difficulty_adjusted_ability"] = (
            out["student_ability_score"]
            / (df["difficulty_level"] * 0.5 + 0.5)
        )

        # Ratio features 
        # 11. accuracy_ratio
        attempted_safe = df["questions_attempted"].replace(0, 1)
        out["accuracy_ratio"] = (
            df["questions_correct"] / attempted_safe
        ).clip(0, 1)

        # 12. consistency_score – higher = more consistent
        out["consistency_score"] = 1.0 / (1.0 + out["learning_consistency"])

        # 13. normalized_time – time per difficulty unit
        diff_safe = df["difficulty_level"].replace(0, 1)
        out["normalized_time"] = df["time_spent_minutes"] / diff_safe

        # Final sanity 
        # Replace any remaining NaN / inf with 0
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        out.fillna(0, inplace=True)

        assert list(out.columns) == self.FEATURE_NAMES, (
            f"Expected {self.FEATURE_NAMES}, got {list(out.columns)}"
        )
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: fit + transform in one call (use on training set)."""
        return self.fit(df).transform(df)

    def get_feature_names(self) -> list:
        """Return the list of all 13 engineered feature names."""
        return list(self.FEATURE_NAMES)


# Quick smoke-test
if __name__ == "__main__":
    from data_preprocessing import load_data, clean_data, split_data

    df = load_data()
    df = clean_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    fe = FeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_val_fe = fe.transform(X_val)
    X_test_fe = fe.transform(X_test)

    print(f"\nEngineered features ({len(fe.get_feature_names())}):")
    for i, name in enumerate(fe.get_feature_names(), 1):
        print(f"  {i:2d}. {name}")

    print(f"\nTrain shape : {X_train_fe.shape}")
    print(f"Val shape   : {X_val_fe.shape}")
    print(f"Test shape  : {X_test_fe.shape}")
    print(X_train_fe.describe().round(2))
    print("\n[OK] Feature engineering complete.")
