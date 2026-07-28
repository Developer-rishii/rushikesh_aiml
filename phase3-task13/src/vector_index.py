"""
Stage B: vector index over resume embeddings.

DESIGN DECISION: at this corpus size (tens-hundreds of resumes) an exact
brute-force cosine search is both correct and fast (<5ms), so it is used
here directly. The class exposes the same `add` / `search` interface a
FAISS `IndexFlatIP` or pgvector table would, so swapping in a real ANN
backend at marketplace scale is a drop-in replacement, not a rewrite -
this is the "pgvector vs a dedicated vector database" alternative named
in the study guide's Section 8, made concrete as an interface boundary.
"""
from __future__ import annotations
import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[str] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def add(self, ids: list[str], vectors: np.ndarray):
        assert vectors.shape[1] == self.dim
        self.ids.extend(ids)
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        if self.vectors.shape[0] == 0:
            return []
        # vectors are pre-normalized -> dot product = cosine similarity
        scores = self.vectors @ query_vec.reshape(-1)
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.ids[i], float(scores[i])) for i in top_idx]

    def __len__(self):
        return len(self.ids)
