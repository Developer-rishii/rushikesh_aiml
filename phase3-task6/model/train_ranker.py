"""
Learning-to-rank model, trained on the REAL logged impressions (with implicit
labels derived from joined outcome events), not on the latent true_relevance
(that would be cheating -- true_relevance is only used later, separately, to
compute an "oracle ceiling" for context).

ALTERNATIVE APPROACHES CONSIDERED (Section 8 of study guide):
  - LightGBM LambdaMART (listwise) was the first choice for a proper
    learning-to-rank objective, but this environment has no network access
    to install it. REJECTED for environment reasons, not correctness.
  - Chose: scikit-learn GradientBoostingClassifier as a POINTWISE ranker
    (predict P(click|item) per candidate, then sort by score). This is a
    real, defensible LTR approach (pointwise), just not the state of the
    art listwise one. Documented here so the trade-off is explicit rather
    than hidden.
  - Client-side vs server-side impression logging: chose server-side
    (implemented in eventlog/ranked_list_logger.py) because it is the only
    way to GUARANTEE completeness (client-side logging silently drops rows
    on tab-close / ad-blockers) - accuracy is sacrificed slightly (server
    doesn't know if the item actually rendered on screen) for completeness,
    which the Definition of Done requires ("outcomes joinable to
    impressions" -- you cannot join what was never logged).
"""
import pandas as pd
import numpy as np
import json
import os
import sys
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.simulate_logs import FEATURE_COLS

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def join_impressions_with_outcomes(event_log_path: str, features_path: str) -> pd.DataFrame:
    """Stage E verification: prove impressions join to outcomes, then build
    the training label from that join (label=1 if the impression got a
    click)."""
    events = pd.read_csv(event_log_path)
    impressions = events[events.event_type == "impression"]
    clicks = events[events.event_type == "click"]

    join_rate = clicks.impression_id.isin(impressions.event_id).mean() if len(clicks) else 1.0
    assert join_rate == 1.0, f"outcome->impression join broken: {join_rate:.3f} joinable"

    clicked_impression_ids = set(clicks.impression_id)
    feats = pd.read_csv(features_path)
    feats["label"] = feats.impression_id.isin(clicked_impression_ids).astype(int)
    return feats, join_rate


def train(feats: pd.DataFrame):
    feats = feats.copy()
    feats["group"] = feats["job_id"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
    train_idx, test_idx = next(gss.split(feats, groups=feats["group"]))
    train_df, test_df = feats.iloc[train_idx], feats.iloc[test_idx]

    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["label"]

    model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.08, random_state=7)
    model.fit(X_train, y_train)

    test_df = test_df.copy()
    test_df["pred_score"] = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_df["pred_score"])
    return model, train_df, test_df, auc


if __name__ == "__main__":
    feats, join_rate = join_impressions_with_outcomes(
        os.path.join(ARTIFACTS, "event_log.csv"),
        os.path.join(ARTIFACTS, "impressions_with_features.csv"),
    )
    model, train_df, test_df, auc = train(feats)
    test_df.to_csv(os.path.join(ARTIFACTS, "test_predictions.csv"), index=False)

    import joblib
    joblib.dump(model, os.path.join(ARTIFACTS, "ranker_model.joblib"))

    summary = {
        "join_rate_outcomes_to_impressions": join_rate,
        "n_train_rows": len(train_df),
        "n_test_rows": len(test_df),
        "test_click_auc": auc,
        "positive_rate_train": float(train_df["label"].mean()),
        "positive_rate_test": float(test_df["label"].mean()),
    }
    with open(os.path.join(ARTIFACTS, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
