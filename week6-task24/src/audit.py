"""
Fairness ceiling: a model trained on the tier-independent, merit-only
`fair_recommended` label, using ONLY merit features (no tier, no historical
bias). This is NOT deployable -- we don't have access to a bias-free ground
truth in real life -- it's a reference bound for how far mitigation can
honestly be pushed on this dataset, so the sign-off can't pretend a
mitigated model is worse than it actually is relative to what's achievable.
"""
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from config import MERIT_FEATURES, PROTECTED_ATTR, SEED
from metrics import disparate_impact


def fairness_ceiling(df: pd.DataFrame, seed: int = SEED) -> dict:
    X = df[MERIT_FEATURES]
    y = df["fair_recommended"]
    X_train, X_test, y_train, y_test, tier_train, tier_test = train_test_split(
        X, y, df[PROTECTED_ATTR], test_size=0.3, random_state=seed, stratify=y
    )
    model = GradientBoostingClassifier(random_state=seed, n_estimators=150, max_depth=3)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return disparate_impact(preds, tier_test.values)


if __name__ == "__main__":
    from config import CANDIDATES_PATH
    from data_validation import clean
    df = clean(pd.read_csv(CANDIDATES_PATH))
    print(fairness_ceiling(df))
