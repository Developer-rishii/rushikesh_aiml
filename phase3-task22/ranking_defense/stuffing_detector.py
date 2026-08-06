"""
stuffing_detector.py
Hybrid defence for Stage C: rule-based signals (fast, explainable) feeding a
supervised classifier (generalizes to unseen stuffing patterns).
"""
import re, math, json, os
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

ZERO_WIDTH = "\u200b"


def rule_signals(text: str) -> dict:
    words = re.findall(r"[a-zA-Z\+\.]+", text.lower())
    n = max(1, len(words))
    counts = Counter(words)
    top_word, top_count = counts.most_common(1)[0] if counts else ("", 0)
    repetition_rate = top_count / n
    unique_ratio = len(counts) / n
    has_hidden_chars = ZERO_WIDTH in text
    # char-level entropy: stuffed/hidden text has abnormal entropy vs prose
    char_counts = Counter(text)
    total = max(1, len(text))
    entropy = -sum((c / total) * math.log2(c / total) for c in char_counts.values())
    return {
        "repetition_rate": repetition_rate,
        "unique_ratio": unique_ratio,
        "has_hidden_chars": float(has_hidden_chars),
        "char_entropy": entropy,
        "length": len(text),
    }


def featurize(texts):
    feats = [rule_signals(t) for t in texts]
    keys = ["repetition_rate", "unique_ratio", "has_hidden_chars", "char_entropy", "length"]
    return np.array([[f[k] for k in keys] for f in feats]), keys


def train_and_eval(candidates, out_dir):
    texts = [c["resume_text"] for c in candidates]
    labels = np.array([1 if c["is_adversarial"] else 0 for c in candidates])
    X, feat_names = featurize(texts)

    X_train, X_test, y_train, y_test, cand_train, cand_test = train_test_split(
        X, labels, candidates, test_size=0.3, random_state=42, stratify=labels
    )

    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(X_train, y_train)

    # Baseline to beat: pure rule threshold (repetition_rate > 0.15 OR hidden chars)
    baseline_pred = np.array([
        1 if (rs[0] > 0.15 or rs[2] > 0) else 0
        for rs in X_test
    ])
    model_pred = clf.predict(X_test)

    def metrics(y_true, y_pred):
        return {
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

    result = {
        "n_test": int(len(y_test)),
        "n_positive_test": int(y_test.sum()),
        "baseline_rule_only": metrics(y_test, baseline_pred),
        "trained_classifier": metrics(y_test, model_pred),
        "feature_names": feat_names,
        "feature_importance": {
            feat_names[i]: round(float(clf.coef_[0][i]), 4) for i in range(len(feat_names))
        },
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stuffing_detector_eval.json"), "w") as f:
        json.dump(result, f, indent=2)

    import joblib
    joblib.dump(clf, os.path.join(out_dir, "stuffing_clf.joblib"))
    return result, clf


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(base, "data", "candidates.json")) as f:
        candidates = json.load(f)
    res, _ = train_and_eval(candidates, os.path.dirname(__file__))
    print(json.dumps(res, indent=2))
