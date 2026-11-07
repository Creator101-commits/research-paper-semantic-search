"""Hybrid retrieval via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from collections import defaultdict

from paper_search.data.schema import Document, SearchResult
from paper_search.retrievers.bm25 import BM25Retriever
from paper_search.retrievers.dense import DenseRetriever


class HybridRRFRetriever:
    """Hybrid rank-fusion retriever combining BM25 lexical and Dense semantic search.

    Parameters
    ----------
    k_rrf : float, default=60.0
        Smoothing constant used in Reciprocal Rank Fusion.
    lexical_weight : float, default=1.0
        Relative weight for BM25 ranking scores.
    dense_weight : float, default=1.0
        Relative weight for Dense ranking scores.
    """

    def __init__(
        self,
        k_rrf: float = 60.0,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
    ) -> None:
        if k_rrf < 0 or lexical_weight < 0 or dense_weight < 0:
            raise ValueError("RRF smoothing and weights must be non-negative.")
        self.k_rrf = k_rrf
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.is_indexed: bool = False

    def index(self, documents: list[Document]) -> HybridRRFRetriever:
        """Index documents into both lexical and dense backends."""
        self.bm25.index(documents)
        self.dense.index(documents)
        self.is_indexed = True
        return self

    def search(
        self,
        query: str,
        k: int = 10,
        candidate_depth: int = 50,
    ) -> list[SearchResult]:
        """Perform hybrid reciprocal rank fusion retrieval."""
        if k <= 0:
            raise ValueError("Result count must be positive.")
        if not self.is_indexed:
            raise RuntimeError("Hybrid retriever is not indexed. Call 'index' first.")

        candidate_depth = max(candidate_depth, k)
        bm25_results = self.bm25.search(query, k=candidate_depth)
        dense_results = self.dense.search(query, k=candidate_depth)

        fused_scores: dict[str, float] = defaultdict(float)

        for res in bm25_results:
            fused_scores[res.doc_id] += self.lexical_weight / (self.k_rrf + res.rank)

        for res in dense_results:
            fused_scores[res.doc_id] += self.dense_weight / (self.k_rrf + res.rank)

        # Sort descending by fused score, deterministic tie-breaking by doc_id
        ranked = sorted(
            fused_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:k]

        return [
            SearchResult(
                doc_id=doc_id,
                score=float(score),
                rank=r + 1,
            )
            for r, (doc_id, score) in enumerate(ranked)
        ]
