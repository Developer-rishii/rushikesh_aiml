"""
evaluation/imbalance_check.py — Step 5: check behaviour under class
imbalance. Directly answers the brainstorming question "Does accuracy
lie because of imbalance?" with a computed comparison, not a guess.
"""
import pandas as pd


def check_imbalance(y_train, y_val) -> dict:
    train_rates = y_train.value_counts(normalize=True).sort_index()
    val_rates = y_val.value_counts(normalize=True).sort_index()
    majority_rate = val_rates.max()

    return {
        "train_class_rates": {str(k): round(float(v), 4) for k, v in train_rates.items()},
        "val_class_rates": {str(k): round(float(v), 4) for k, v in val_rates.items()},
        "val_majority_baseline_accuracy": round(float(majority_rate), 4),
        "accuracy_is_potentially_misleading": bool(majority_rate > 0.6),
        "note": (
            f"A majority-class classifier scores {majority_rate:.1%} accuracy on validation "
            "by predicting nothing meaningful — accuracy alone cannot be trusted here; "
            "confusion matrix + precision/recall + the cost-based threshold are the "
            "metrics actually reported and acted on in this project."
        ),
    }
