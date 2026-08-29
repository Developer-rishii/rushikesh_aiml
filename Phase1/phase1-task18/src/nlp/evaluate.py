"""
nlp/evaluate.py — Step 4: evaluate with a task-appropriate metric.
Multi-class text classification -> macro-F1 (treats every category
equally) reported alongside per-class precision/recall/F1 and a full
confusion matrix — never accuracy alone, which is the direct guard
against "wrong metric for the task" on a task with 6 balanced-but-
distinct categories where per-category performance matters.
"""
import logging
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

log = logging.getLogger("src.nlp.evaluate")


def evaluate_classifier(y_true, y_pred, labels: list) -> dict:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    accuracy = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    result = {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "per_class_report": {k: {m: round(v, 4) if isinstance(v, float) else v for m, v in vals.items()}
                              for k, vals in report.items() if k in labels},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
    }
    log.info("[Step 4] macro_f1=%.4f, accuracy=%.4f (macro_f1 is the primary metric — "
              "accuracy alone would hide any single weak category)", macro_f1, accuracy)
    return result
