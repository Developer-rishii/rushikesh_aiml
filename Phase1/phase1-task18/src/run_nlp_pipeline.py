"""
run_nlp_pipeline.py — Task 18's full flow, in the study guide's exact
step order:
  1. Clean and tokenise the text data.
  2. Vectorise with TF-IDF or embeddings.
  3. Build the target NLP task (classify/extract/match).
  4. Evaluate with a task-appropriate metric.
  5. Inspect errors for language-specific failure modes.
  6. Package the text pipeline for reuse.

Run: python -m src.run_nlp_pipeline
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split

from configs.loader import load_config
from src.nlp.clean_text import clean_corpus
from src.nlp.vectorize import build_bow_pipeline, build_lsa_pipeline
from src.nlp.evaluate import evaluate_classifier
from src.nlp.error_inspection import find_worst_errors, summarize_confusions
from src.nlp.package import package_for_reuse, load_packaged_pipeline, predict_category

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("src.run_nlp_pipeline")


def main():
    t0 = time.time()
    cfg = load_config()
    log.info("Loaded config: %s", cfg)

    if not cfg.corpus_path.exists():
        log.error("Corpus not found at %s. Run data/generate_corpus.py first.", cfg.corpus_path)
        sys.exit(1)
    df = pd.read_csv(cfg.corpus_path)
    if df.empty:
        log.error("Corpus is empty.")
        sys.exit(1)

    try:
        df_clean, n_empty = clean_corpus(df, cfg)
    except ValueError as e:
        log.error("Text cleaning failed: %s", e)
        sys.exit(1)
    if n_empty > 0:
        df_clean = df_clean[df_clean["clean_text"].str.len() > 0].reset_index(drop=True)
        log.warning("Dropped %s document(s) that were empty after cleaning.", n_empty)

    labels = sorted(df_clean[cfg.label_col].unique().tolist())

    X_temp, X_test, y_temp, y_test = train_test_split(
        df_clean["clean_text"], df_clean[cfg.label_col],
        test_size=cfg.test_frac, stratify=df_clean[cfg.label_col], random_state=cfg.seed,
    )
    rel_val = cfg.val_frac / (1 - cfg.test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=rel_val, stratify=y_temp, random_state=cfg.seed,
    )
    log.info("Split: train=%s val=%s test=%s", len(X_train), len(X_val), len(X_test))

    try:
        bow_pipeline = build_bow_pipeline(cfg)
        bow_pipeline.fit(X_train, y_train)
        lsa_pipeline = build_lsa_pipeline(cfg)
        lsa_pipeline.fit(X_train, y_train)
    except ValueError as e:
        log.error("Pipeline build/fit failed: %s", e)
        sys.exit(1)

    bow_val_pred = bow_pipeline.predict(X_val)
    lsa_val_pred = lsa_pipeline.predict(X_val)
    bow_val_eval = evaluate_classifier(y_val, bow_val_pred, labels)
    lsa_val_eval = evaluate_classifier(y_val, lsa_val_pred, labels)

    winner_name = "tfidf_bow" if bow_val_eval["macro_f1"] >= lsa_val_eval["macro_f1"] else "tfidf_lsa"
    winner_pipeline = bow_pipeline if winner_name == "tfidf_bow" else lsa_pipeline
    log.info("Validation macro-F1 — bag-of-words: %.4f | LSA (meaning-aware): %.4f -> winner: %s",
              bow_val_eval["macro_f1"], lsa_val_eval["macro_f1"], winner_name)

    test_pred = winner_pipeline.predict(X_test)
    test_proba = winner_pipeline.predict_proba(X_test)
    test_proba_max = test_proba.max(axis=1)
    test_eval = evaluate_classifier(y_test, test_pred, labels)

    worst_errors = find_worst_errors(X_test.tolist(), y_test.tolist(), test_pred.tolist(),
                                       test_proba_max.tolist(), cfg.n_errors_to_inspect)
    confusion_summary = summarize_confusions(worst_errors)

    # ---- Step 5 (continued): stress-test on deliberately AMBIGUOUS hybrid
    # documents (mixing vocabulary from two related categories) — the
    # clean primary corpus above is well-separated by design (6 distinct
    # professions), so it alone can't demonstrate what error inspection
    # looks like on a genuinely hard case. This is documented, not hidden. ----
    stress_path = cfg.corpus_path.parent / "stress_test_hybrids.csv"
    stress_result = None
    if stress_path.exists():
        from src.nlp.clean_text import clean_text as _clean_one
        hybrids = pd.read_csv(stress_path)
        hybrids["clean_text"] = hybrids["text"].apply(lambda t: _clean_one(t, cfg))
        hybrid_pred = winner_pipeline.predict(hybrids["clean_text"])
        hybrid_proba = winner_pipeline.predict_proba(hybrids["clean_text"]).max(axis=1)
        hybrid_eval = evaluate_classifier(hybrids["category"], hybrid_pred, labels)
        hybrid_errors = find_worst_errors(hybrids["text"].tolist(), hybrids["category"].tolist(),
                                            hybrid_pred.tolist(), hybrid_proba.tolist(), cfg.n_errors_to_inspect)
        hybrid_confusion = summarize_confusions(hybrid_errors)
        hybrid_errors.to_csv(cfg.report_dir / "stress_test_worst_errors.csv", index=False)
        stress_result = {
            "n_hybrid_documents": len(hybrids),
            "macro_f1_on_ambiguous_hybrids": hybrid_eval["macro_f1"],
            "accuracy_on_ambiguous_hybrids": hybrid_eval["accuracy"],
            "confusion_summary": hybrid_confusion,
        }
        log.info("[Step 5] Stress test on %s deliberately ambiguous hybrid documents: "
                  "macro_f1=%.4f (real errors: %s) — %s",
                  len(hybrids), hybrid_eval["macro_f1"], hybrid_confusion["n_errors"], hybrid_confusion["note"])

    cfg.report_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.figure_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "winner": winner_name,
        "labels": labels,
        "test_macro_f1": test_eval["macro_f1"],
        "seed": cfg.seed,
    }
    package_paths = package_for_reuse(winner_pipeline, cfg, metadata, cfg.artifact_dir)

    reloaded_model, reloaded_cfg_dict = load_packaged_pipeline(cfg.artifact_dir)
    sample_text = df.iloc[0][cfg.text_col]
    live_prediction = predict_category(winner_pipeline, cfg, sample_text)
    reloaded_prediction = predict_category(reloaded_model, cfg, sample_text)
    package_verified = live_prediction == reloaded_prediction
    log.info("[Step 6] Packaged pipeline reload check: predictions match = %s", package_verified)

    worst_errors.to_csv(cfg.report_dir / "worst_errors.csv", index=False)

    result = {
        "seed": cfg.seed,
        "corpus_size": len(df),
        "n_categories": len(labels),
        "categories": labels,
        "n_documents_dropped_empty_after_cleaning": n_empty,
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "bow_val_macro_f1": bow_val_eval["macro_f1"],
        "lsa_val_macro_f1": lsa_val_eval["macro_f1"],
        "winner_representation": winner_name,
        "test_evaluation": test_eval,
        "error_inspection": confusion_summary,
        "stress_test_ambiguous_hybrids": stress_result,
        "package_paths": package_paths,
        "package_reload_verified": package_verified,
        "example_prediction": live_prediction,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (cfg.report_dir / "nlp_pipeline_report.json").write_text(json.dumps(result, indent=2, default=str))
    (cfg.log_dir / "run_nlp_pipeline.log").write_text(json.dumps(result, indent=2, default=str))

    log.info("Done in %ss. Winner=%s, test macro-F1=%.4f. Report -> %s",
              result["runtime_seconds"], winner_name, test_eval["macro_f1"],
              cfg.report_dir / "nlp_pipeline_report.json")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
