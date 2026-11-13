"""Hybrid retrieval: BM25 and dense rankings fused with Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Callable

import bm25s
import numpy as np


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Each ranking contributes 1/(k + rank) per doc; docs ranked well by several lists win."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, bm25: bm25s.BM25, vectors: np.ndarray, embed_query: Callable[[str], np.ndarray]):
        self.bm25, self.vectors, self.embed_query = bm25, vectors, embed_query

    def _bm25(self, text: str, depth: int) -> list[int]:
        depth = min(depth, self.vectors.shape[0])
        docs, scores = self.bm25.retrieve(
            bm25s.tokenize([text], stopwords="en", show_progress=False), k=depth, show_progress=False
        )
        return [int(d) for d, s in zip(docs[0], scores[0]) if s > 0]

    def _dense(self, text: str, depth: int) -> list[int]:
        sims = self.vectors.astype(np.float32) @ self.embed_query(text)
        return np.argsort(-sims)[:depth].tolist()

    def search(self, text: str, k: int = 50, depth: int = 100, mode: str = "hybrid") -> list[tuple[int, float]]:
        if mode == "bm25":
            return [(d, 1.0 / (r + 1)) for r, d in enumerate(self._bm25(text, depth))][:k]
        if mode == "dense":
            return [(d, 1.0 / (r + 1)) for r, d in enumerate(self._dense(text, depth))][:k]
        if mode == "hybrid":
            return rrf([self._bm25(text, depth), self._dense(text, depth)])[:k]
        raise ValueError(f"unknown mode {mode!r}")
