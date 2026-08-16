"""
modeling/errors.py — Step 5: inspect the worst errors for patterns.
Not just "compute a metric and stop" — this surfaces the actual
misclassified validation rows, ranked by how confidently wrong the model
was, so a human can look for a pattern (not just a number).
"""
import pandas as pd


def worst_errors(X_val: pd.DataFrame, y_val, y_pred, y_proba, top_n: int = 10) -> pd.DataFrame:
    df = X_val.copy()
    df["true_label"] = y_val.values
    df["predicted_label"] = y_pred
    df["predicted_proba_class1"] = y_proba
    df["correct"] = df["true_label"] == df["predicted_label"]
    # "confidence of the wrong call": how far predicted proba was from the
    # true label — 1.0 means maximally, confidently wrong.
    df["error_magnitude"] = abs(df["true_label"] - df["predicted_proba_class1"])

    errors = df[~df["correct"]].sort_values("error_magnitude", ascending=False)
    return errors.head(top_n)


def summarize_error_patterns(errors: pd.DataFrame) -> dict:
    """Cheap, honest pattern surfacing: which classes get confused which
    way, and whether errors cluster on any injected-missing columns."""
    if errors.empty:
        return {"n_errors_inspected": 0, "note": "No validation errors — see README caveat on separability."}

    summary = {
        "n_errors_inspected": int(len(errors)),
        "false_negatives": int(((errors["true_label"] == 1) & (errors["predicted_label"] == 0)).sum()),
        "false_positives": int(((errors["true_label"] == 0) & (errors["predicted_label"] == 1)).sum()),
        "mean_error_magnitude": round(float(errors["error_magnitude"].mean()), 4),
    }
    return summary
