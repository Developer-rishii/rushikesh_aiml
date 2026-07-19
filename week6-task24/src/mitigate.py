"""
Bias mitigation, built on a real trained model (GradientBoostingClassifier,
not a toy linear rule), two stages:

1. Reweighing (Kamiran & Calders, 2012): training samples are weighted so
   group and label are statistically independent in the *weighted* training
   distribution. `college_tier` is also dropped from the feature set
   (fairness through unawareness) so the model has no direct handle on it
   at inference time.

2. Per-group threshold post-processing: even after reweighing, finite
   samples can leave positive-prediction rates slightly uneven across
   groups. We search, per group, for the probability threshold that brings
   the group's positive rate in line with the overall target rate, subject
   to the disparate-impact floor.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from config import MERIT_FEATURES, PROTECTED_ATTR, LABEL_COL, DI_TARGET
from metrics import disparate_impact


def compute_reweighing_weights(y: pd.Series, group: pd.Series) -> np.ndarray:
    n = len(y)
    w = np.ones(n)
    for g in group.unique():
        for label_val in y.unique():
            mask = (group == g) & (y == label_val)
            n_gl = mask.sum()
            if n_gl == 0:
                continue
            p_g = (group == g).mean()
            p_l = (y == label_val).mean()
            expected = p_g * p_l * n
            w[mask.values] = expected / n_gl
    return w


def find_group_thresholds(probs: np.ndarray, group: np.ndarray, target_rate: float,
                           di_floor: float = DI_TARGET, grid=np.linspace(0.05, 0.95, 181)):
    groups = sorted(pd.unique(group))
    thresholds = {}
    for g in groups:
        mask = group == g
        gp = probs[mask]
        best_t, best_diff = 0.5, np.inf
        for t in grid:
            rate = (gp >= t).mean()
            diff = abs(rate - target_rate)
            if diff < best_diff:
                best_diff, best_t = diff, t
        thresholds[g] = best_t

    def group_rates(th):
        return {g: (probs[group == g] >= th[g]).mean() for g in groups}

    rates = group_rates(thresholds)
    for _ in range(400):
        valid = [r for r in rates.values() if r is not None]
        di = min(valid) / max(valid) if max(valid) > 0 else 0.0
        if di >= di_floor:
            break
        worst_g = min(rates, key=rates.get)
        best_g = max(rates, key=rates.get)
        if thresholds[worst_g] > grid.min():
            thresholds[worst_g] = max(grid.min(), thresholds[worst_g] - 0.005)
        if thresholds[best_g] < grid.max():
            thresholds[best_g] = min(grid.max(), thresholds[best_g] + 0.005)
        rates = group_rates(thresholds)
    return thresholds


def train_mitigated(df: pd.DataFrame, seed: int = 42):
    """Returns (model, thresholds, split_data, audit_result). split_data is
    (X_test, y_test, tier_test, preds, df_test) -- df_test carries the full
    row (including fair_recommended) so callers can evaluate quality against
    both the historical (biased) label and the merit ground truth."""
    X = df[MERIT_FEATURES]
    y = df[LABEL_COL]
    tier = df[PROTECTED_ATTR]

    train_idx, test_idx = train_test_split(
        df.index, test_size=0.3, random_state=seed, stratify=y
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    tier_train, tier_test = tier.loc[train_idx], tier.loc[test_idx]

    weights = compute_reweighing_weights(y_train.reset_index(drop=True),
                                          tier_train.reset_index(drop=True))
    model = GradientBoostingClassifier(random_state=seed, n_estimators=150, max_depth=3)
    model.fit(X_train, y_train, sample_weight=weights)

    probs_test = model.predict_proba(X_test)[:, 1]
    overall_target_rate = y_train.mean()

    thresholds = find_group_thresholds(probs_test, tier_test.values, overall_target_rate)
    preds = np.array([
        1 if p >= thresholds[g] else 0
        for p, g in zip(probs_test, tier_test.values)
    ])

    audit_result = disparate_impact(preds, tier_test.values)
    split_data = (X_test, y_test.values, tier_test.values, preds, df.loc[test_idx])
    return model, thresholds, split_data, audit_result


if __name__ == "__main__":
    from config import CANDIDATES_PATH
    from data_validation import clean
    df = clean(pd.read_csv(CANDIDATES_PATH))
    _, thresholds, _, audit_result = train_mitigated(df)
    print("Per-group thresholds:", thresholds)
    print("Mitigated audit:", audit_result)
