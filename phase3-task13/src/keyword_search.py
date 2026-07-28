"""
Keyword baseline: BM25, implemented from scratch (no `rank_bm25` package
available offline in this environment). This is the baseline every
Stage-C deliverable must be evaluated against ("the baseline you must
beat"), and it is also half of the hybrid retriever in Stage D.
"""
from __future__ import annotations
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#]*")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self.doc_freqs: list[Counter] = []
        self.doc_lens: list[int] = []
        self.avg_len = 0.0
        self.df: Counter = Counter()
        self.N = 0

    def fit(self, ids: list[str], docs: list[str]):
        self.ids = ids
        for d in docs:
            toks = tokenize(d)
            self.doc_freqs.append(Counter(toks))
            self.doc_lens.append(len(toks))
            for t in set(toks):
                self.df[t] += 1
        self.N = len(docs)
        self.avg_len = sum(self.doc_lens) / max(self.N, 1)
        return self

    def _idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        q_toks = tokenize(query)
        scores = [0.0] * self.N
        for i in range(self.N):
            dl = self.doc_lens[i]
            freqs = self.doc_freqs[i]
            s = 0.0
            for t in q_toks:
                f = freqs.get(t, 0)
                if f == 0:
                    continue
                idf = self._idf(t)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                s += idf * (f * (self.k1 + 1)) / denom
            scores[i] = s
        order = sorted(range(self.N), key=lambda i: -scores[i])[:top_k]
        return [(self.ids[i], scores[i]) for i in order]
