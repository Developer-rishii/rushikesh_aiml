"""
Simulates the Task 21 hand-off: trains the ORIGINAL unmitigated model
(college_tier used directly as a feature, trained on the historically
biased label) and persists its audit findings as a versioned artifact.

Task 24 (sign_off.py) treats this file as an upstream dependency it must
load and validate -- exactly like a real pipeline would consume another
team's hand-off -- rather than silently recomputing it inline.
"""
import json
import datetime as dt

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import (CANDIDATES_PATH, TASK21_AUDIT_PATH, MERIT_FEATURES,
                     PROTECTED_ATTR, LABEL_COL, SEED)
from data_validation import clean
from metrics import disparate_impact, classification_report


def run(seed: int = SEED, out_path: str = TASK21_AUDIT_PATH) -> dict:
    df = clean(pd.read_csv(CANDIDATES_PATH))

    features = MERIT_FEATURES + [PROTECTED_ATTR]
    X, y = df[features], df[LABEL_COL]
    X_train, X_test, y_train, y_test, tier_train, tier_test = train_test_split(
        X, y, df[PROTECTED_ATTR], test_size=0.3, random_state=seed, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    audit = disparate_impact(preds, tier_test.values)
    quality = classification_report(y_test.values, preds)

    result = {
        "artifact": "TASK21_FAIRNESS_AUDIT_RESULTS",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_evaluated": "baseline unmitigated LogisticRegression (college_tier used as a feature)",
        "protected_attribute": PROTECTED_ATTR,
        "disparate_impact": audit["disparate_impact"],
        "group_positive_rates": audit["group_rates"],
        "quality_metrics": quality,
        "finding": (
            "FAIL" if audit["disparate_impact"] < 0.80 else "PASS"
        ),
        "notes": (
            "Model's positive-recommendation rate varies sharply by college_tier "
            "despite skill_score being statistically independent of tier in the "
            "underlying data. This indicates the historical label (and therefore "
            "the trained model) encodes a prestige proxy bias, not a merit "
            "difference."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
