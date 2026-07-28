"""
Stage B: "An embedding + vector index over resumes/JDs"
--------------------------------------------------------
DESIGN DECISION (documented per Stage A.3 - "write down WHY, including
what you rejected"):

  Rejected: sentence-transformers / a hosted embedding API.
  Reason rejected: this environment has no outbound network access, so
  neither a pip install of sentence-transformers nor a live API call to
  an embedding endpoint is possible here. Shipping code that silently
  can't run is worse than shipping a real, working, honestly-labelled
  alternative.

  Chosen: TF-IDF -> Truncated SVD (i.e. classic LSA), a genuine
  distributional-semantics embedding technique that captures
  co-occurrence structure (so "data pipelines" ends up near "ETL",
  "Airflow", "Spark jobs" because they co-occur across the corpus),
  not just literal token overlap. This is a legitimate "embedding"
  under the study guide's own definition ("text mapped into vectors
  where distance means semantic similarity") and needs no external
  service, which also lets us demo the "model unavailable" failure mode
  honestly (Stage E.3) without faking it.

  In a real deployment with network access, swap `Embedder.fit_transform`
  / `Embedder.encode` for a sentence-transformers or API-based encoder -
  the rest of the pipeline (index, hybrid, eval) is agnostic to that choice.
"""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import pickle
import os
import logging

class Embedder:
    def __init__(self, n_components: int = 64, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.use_st = os.environ.get("USE_SENTENCE_TRANSFORMERS", "0") == "1"
        self._fitted = False
        
        if self.use_st:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                logging.warning("sentence-transformers not available, falling back to LSA")
                self.use_st = False

        if not self.use_st:
            self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)

    def fit(self, corpus: list[str]):
        if self.use_st:
            self._fitted = True
            return self

        # Edge case: max_df=0.9 is meaningless (and raises) on tiny/single-doc
        # corpora, so relax it when the corpus is too small for the ratio to apply.
        max_df = 0.9 if len(corpus) >= 3 else 1.0
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_df=max_df, sublinear_tf=True
        )
        tfidf = self.vectorizer.fit_transform(corpus)
        # SVD components can't exceed min(n_samples, n_features)-1
        k = min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if k != self.svd.n_components:
            self.svd = TruncatedSVD(n_components=max(k, 2), random_state=42)
        self.svd.fit(tfidf)
        self._fitted = True
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder must be fit() before encode().")
        if self.use_st:
            vecs = self.model.encode(texts)
            return normalize(vecs)

        tfidf = self.vectorizer.transform(texts)
        vecs = self.svd.transform(tfidf)
        return normalize(vecs)  # L2-normalize so dot product == cosine similarity

    def fit_transform(self, corpus: list[str]) -> np.ndarray:
        self.fit(corpus)
        return self.encode(corpus)

    def save(self, path: str):
        model_tmp = getattr(self, "model", None)
        if hasattr(self, "model"):
            self.model = None
        with open(path, "wb") as f:
            pickle.dump(self, f)
        if model_tmp is not None:
            self.model = model_tmp

    @staticmethod
    def load(path: str) -> "Embedder":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if getattr(obj, "use_st", False):
            from sentence_transformers import SentenceTransformer
            obj.model = SentenceTransformer('all-MiniLM-L6-v2')
        return obj
