"""
nlp/vectorize.py — Step 2: vectorise with TF-IDF (word-matching) and
TF-IDF+LSA (meaning-aware, via SVD) — built as two complete, comparable
sklearn Pipelines so Step 3 can plug either into an identical classifier
and Step 4 can compare them head-to-head on identical data.
"""
import logging
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression

log = logging.getLogger("src.nlp.vectorize")

CLASSIFIER_BUILDERS = {
    "logreg": lambda params: LogisticRegression(**params),
}


def build_bow_pipeline(cfg) -> Pipeline:
    """Pure bag-of-words: TF-IDF -> classifier. Word-matching only —
    two documents score as related ONLY if they share literal terms."""
    if cfg.classifier_name not in CLASSIFIER_BUILDERS:
        raise ValueError(f"Unknown classifier '{cfg.classifier_name}'.")
    tfidf = TfidfVectorizer(
        ngram_range=cfg.tfidf_ngram_range,
        max_features=cfg.tfidf_max_features,
        min_df=cfg.tfidf_min_df,
    )
    clf = CLASSIFIER_BUILDERS[cfg.classifier_name](cfg.classifier_params)
    log.info("[Step 2] Built TF-IDF (bag-of-words) pipeline: ngram_range=%s, max_features=%s",
              cfg.tfidf_ngram_range, cfg.tfidf_max_features)
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def build_lsa_pipeline(cfg) -> Pipeline:
    """Meaning-aware: TF-IDF -> Truncated SVD (LSA) -> classifier. SVD
    compresses co-occurring-term patterns into dense components, so two
    documents can score as related even without sharing exact words —
    the property pure TF-IDF structurally lacks."""
    if cfg.classifier_name not in CLASSIFIER_BUILDERS:
        raise ValueError(f"Unknown classifier '{cfg.classifier_name}'.")
    tfidf = TfidfVectorizer(
        ngram_range=cfg.tfidf_ngram_range,
        max_features=cfg.tfidf_max_features,
        min_df=cfg.tfidf_min_df,
    )
    svd = TruncatedSVD(n_components=cfg.lsa_n_components, random_state=cfg.seed)
    clf = CLASSIFIER_BUILDERS[cfg.classifier_name](cfg.classifier_params)
    log.info("[Step 2] Built TF-IDF+LSA (meaning-aware) pipeline: n_components=%s", cfg.lsa_n_components)
    return Pipeline([("tfidf", tfidf), ("svd", svd), ("clf", clf)])
