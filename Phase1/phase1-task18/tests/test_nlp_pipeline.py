"""
Tests for Task 18. One test per named pitfall, plus a live end-to-end run
and edge cases.
Run: python tests/test_nlp_pipeline.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from configs.loader import load_config
from src.nlp.clean_text import clean_text, clean_corpus
from src.nlp.vectorize import build_bow_pipeline, build_lsa_pipeline
from src.nlp.evaluate import evaluate_classifier
from src.nlp.package import load_packaged_pipeline


def test_live_end_to_end_run():
    from src.run_nlp_pipeline import main
    result = main()
    assert result["test_evaluation"]["macro_f1"] > 0
    assert result["package_reload_verified"] is True
    print(f"PASS: live end-to-end run — winner={result['winner_representation']}, "
          f"test macro-F1={result['test_evaluation']['macro_f1']}")


def test_pitfall_text_cleaning_actually_runs():
    """Pitfall: Skipping text cleaning."""
    cfg = load_config()
    raw = "The Data-Scientist role requires PYTHON, SQL, and Statistics!!! Apply NOW."
    cleaned = clean_text(raw, cfg)
    assert cleaned == cleaned.lower(), "output is not lowercased — cleaning did not run"
    assert "!!!" not in cleaned and "," not in cleaned, "punctuation was not stripped"
    assert "the" not in cleaned.split() and "and" not in cleaned.split(), "stopwords were not removed"
    print(f"PASS: cleaning demonstrably transforms text — {raw!r} -> {cleaned!r}")


def test_pitfall_bow_vs_meaning_actually_compared():
    """Pitfall: Bag-of-words where meaning matters."""
    cfg = load_config()
    df = pd.read_csv(cfg.corpus_path)
    df_clean, _ = clean_corpus(df, cfg)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        df_clean["clean_text"], df_clean[cfg.label_col], test_size=0.2,
        stratify=df_clean[cfg.label_col], random_state=cfg.seed,
    )
    bow = build_bow_pipeline(cfg)
    bow.fit(X_train, y_train)
    lsa = build_lsa_pipeline(cfg)
    lsa.fit(X_train, y_train)

    labels = sorted(df_clean[cfg.label_col].unique().tolist())
    bow_eval = evaluate_classifier(y_val, bow.predict(X_val), labels)
    lsa_eval = evaluate_classifier(y_val, lsa.predict(X_val), labels)
    assert 0 < bow_eval["macro_f1"] <= 1.0
    assert 0 < lsa_eval["macro_f1"] <= 1.0
    print(f"PASS: bag-of-words (macro-F1={bow_eval['macro_f1']}) and meaning-aware LSA "
          f"(macro-F1={lsa_eval['macro_f1']}) both actually fit and evaluated head-to-head, "
          f"not assumed equivalent or skipped")


def test_pitfall_metric_is_not_accuracy_alone():
    """Pitfall: Wrong metric for the task."""
    cfg = load_config()
    assert cfg.primary_metric == "macro_f1", (
        f"primary_metric is '{cfg.primary_metric}' — accuracy alone would be the wrong "
        f"metric for imbalanced-risk multi-class text classification"
    )
    print(f"PASS: primary metric is '{cfg.primary_metric}' (per-class-aware), "
          f"not raw accuracy, for a 6-category classification task")


def test_edge_case_non_string_input_raises():
    cfg = load_config()
    try:
        clean_text(12345, cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: non-string input to the cleaning step raises clearly")


def test_edge_case_empty_corpus_raises():
    cfg = load_config()
    try:
        clean_corpus(pd.DataFrame(columns=[cfg.text_col, cfg.label_col]), cfg)
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("PASS: an empty corpus raises clearly instead of failing deep inside sklearn")


def test_edge_case_missing_package_raises():
    try:
        load_packaged_pipeline(Path("/tmp/does_not_exist_nlp_package"))
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
    print("PASS: loading a missing/incomplete packaged pipeline raises clearly")


if __name__ == "__main__":
    test_pitfall_text_cleaning_actually_runs()
    test_pitfall_metric_is_not_accuracy_alone()
    test_edge_case_non_string_input_raises()
    test_edge_case_empty_corpus_raises()
    test_edge_case_missing_package_raises()
    test_pitfall_bow_vs_meaning_actually_compared()
    test_live_end_to_end_run()
    print("\nALL TESTS PASSED")
