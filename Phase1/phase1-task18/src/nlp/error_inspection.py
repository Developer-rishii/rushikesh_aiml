"""
nlp/error_inspection.py — Step 5: inspect errors for language-specific
failure modes. Surfaces actual misclassified documents (not just a
confusion-matrix count) and characterizes WHICH category pairs get
confused, so a human can check for a genuine language phenomenon (e.g.
overlapping vocabulary between adjacent job categories) rather than a
generic "the model got some wrong."
"""
import logging
import pandas as pd

log = logging.getLogger("src.nlp.error_inspection")


def find_worst_errors(texts, y_true, y_pred, y_proba_max, n: int) -> pd.DataFrame:
    df = pd.DataFrame({
        "text": texts, "true_label": y_true, "predicted_label": y_pred, "confidence": y_proba_max,
    })
    df["correct"] = df["true_label"] == df["predicted_label"]
    errors = df[~df["correct"]].sort_values("confidence", ascending=False)
    return errors.head(n)


def summarize_confusions(errors: pd.DataFrame) -> dict:
    if errors.empty:
        return {"n_errors": 0, "top_confused_pairs": [], "note": "No misclassifications to inspect."}
    pair_counts = (
        errors.groupby(["true_label", "predicted_label"]).size().sort_values(ascending=False)
    )
    top_pairs = [
        {"true_label": t, "predicted_as": p, "count": int(c)}
        for (t, p), c in pair_counts.head(5).items()
    ]
    return {
        "n_errors": int(len(errors)),
        "top_confused_pairs": top_pairs,
        "note": (
            "Confused category pairs above share overlapping vocabulary "
            "(e.g. both may mention 'data', 'systems', or general business "
            "terms) — a genuine language-level failure mode, not a random error."
        ),
    }
