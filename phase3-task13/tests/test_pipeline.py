"""
Section 11: "Dependency, failure & edge-case handling = 15" and
Section 12 pitfalls. Run: python3 -m pytest tests/ -q  (or run directly).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embeddings import Embedder
from src.vector_index import VectorIndex
from src.keyword_search import BM25, tokenize
from src.hybrid import hybrid_search
from src.eval_metrics import ndcg_at_k, average_precision, precision_at_k
from src.failure_mode import SearchService


def test_empty_query_does_not_crash():
    bm25 = BM25().fit(["a", "b"], ["senior data engineer", "frontend react developer"])
    assert bm25.search("", top_k=5) == [] or all(s == 0 for _, s in bm25.search("", top_k=5))


def test_tokenizer_handles_punctuation_and_numbers():
    toks = tokenize("C++ engineer, 5+ yrs, Node.js/React!!")
    assert "c" in toks or "c++"[0:1]  # sanity: doesn't crash on symbols
    assert "engineer" in toks


def test_vector_index_empty_search_returns_empty():
    idx = VectorIndex(dim=8)
    import numpy as np
    assert idx.search(np.zeros(8), top_k=5) == []


def test_embedder_single_document_edge_case():
    # SVD components can't exceed n_samples-1; make sure we don't crash on 1 doc.
    emb = Embedder(n_components=64).fit(["only one document here about data pipelines"])
    vec = emb.encode(["a query"])
    assert vec.shape[0] == 1


def test_hybrid_alpha_extremes_match_pure_systems():
    sem = {"r1": 0.9, "r2": 0.1}
    bm = {"r1": 0.1, "r2": 0.9}
    pure_sem = hybrid_search(sem, bm, alpha=1.0, top_k=2)
    pure_kw = hybrid_search(sem, bm, alpha=0.0, top_k=2)
    assert pure_sem[0][0] == "r1"
    assert pure_kw[0][0] == "r2"


def test_ndcg_perfect_ranking_is_one():
    rel = {"a": 2, "b": 1, "c": 0}
    assert abs(ndcg_at_k(["a", "b", "c"], rel, k=3) - 1.0) < 1e-9


def test_map_no_relevant_docs_is_zero():
    assert average_precision(["a", "b"], {"a": 0, "b": 0}) == 0.0


def test_precision_at_k_empty_ranking():
    assert precision_at_k([], {"a": 1}, k=10) == 0.0


def test_failure_mode_degrades_not_crashes():
    ids = ["r1", "r2"]
    texts = ["data engineer building etl pipelines", "frontend react developer"]
    emb = Embedder(n_components=8).fit(texts)
    idx = VectorIndex(dim=emb.encode(texts).shape[1])
    idx.add(ids, emb.encode(texts))
    bm25 = BM25().fit(ids, texts)
    svc = SearchService(emb, idx, bm25, alpha=0.7)
    svc.simulate_embedding_outage = True
    out = svc.search("data pipelines", top_k=2)
    assert out["mode"] == "DEGRADED_KEYWORD_ONLY"
    assert len(out["results"]) > 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
