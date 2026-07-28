"""
Stage B.4 / Stage E.3: "plus what happens when the model is unavailable"
and "deliberately induce the failure and confirm the designed
degradation actually happens".

The designed behaviour: if the embedding/vector-index path raises
(timeout, OOM, corrupt index, etc.) the search service must not 500 -
it must degrade to BM25-only and say so, because keyword search alone is
still useful and "no results" is a worse outcome for a recruiter than
"slightly worse results".
"""
from __future__ import annotations
from .keyword_search import BM25


class SearchService:
    def __init__(self, embedder, index, bm25: BM25, alpha: float):
        self.embedder = embedder
        self.index = index
        self.bm25 = bm25
        self.alpha = alpha
        self.simulate_embedding_outage = False  # toggled by the demo to induce failure

    def search(self, query: str, top_k: int = 10) -> dict:
        bm25_ranked = self.bm25.search(query, top_k=len(self.bm25.ids))
        bm25_scores = dict(bm25_ranked)

        if self.simulate_embedding_outage:
            ranked = sorted(bm25_scores.items(), key=lambda x: -x[1])[:top_k]
            return {
                "mode": "DEGRADED_KEYWORD_ONLY",
                "warning": "embedding service unavailable - served BM25-only fallback",
                "results": ranked,
            }

        try:
            qvec = self.embedder.encode([query])[0]
            sem_ranked = self.index.search(qvec, top_k=len(self.index.ids))
            sem_scores = dict(sem_ranked)
            from .hybrid import hybrid_search
            ranked = hybrid_search(sem_scores, bm25_scores, self.alpha, top_k=top_k)
            return {"mode": "HYBRID", "warning": None, "results": ranked}
        except Exception as e:  # pragma: no cover - defensive path, exercised by demo
            ranked = sorted(bm25_scores.items(), key=lambda x: -x[1])[:top_k]
            return {
                "mode": "DEGRADED_KEYWORD_ONLY",
                "warning": f"embedding path raised {type(e).__name__}: {e} - served BM25-only fallback",
                "results": ranked,
            }
