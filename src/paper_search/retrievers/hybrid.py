"""Hybrid retrieval via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from collections import defaultdict

from paper_search.data.schema import Document, SearchResult
from paper_search.retrievers.base import Retriever
from paper_search.retrievers.bm25 import BM25Retriever
from paper_search.retrievers.dense import HashedTfidfRetriever


def reciprocal_rank_fusion(
    lexical_results: list[SearchResult],
    dense_results: list[SearchResult],
    *,
    k_rrf: float = 60.0,
    lexical_weight: float = 1.0,
    dense_weight: float = 1.0,
    k: int = 10,
) -> list[SearchResult]:
    """Fuse two ranked lists with weighted Reciprocal Rank Fusion.

    Documents absent from a list contribute nothing from that list. Ties in
    fused score resolve by document ID ascending.
    """
    if k <= 0:
        raise ValueError("Result count must be positive.")
    if k_rrf < 0 or lexical_weight < 0 or dense_weight < 0:
        raise ValueError("RRF smoothing and weights must be non-negative.")

    fused_scores: dict[str, float] = defaultdict(float)

    for res in lexical_results:
        fused_scores[res.doc_id] += lexical_weight / (k_rrf + res.rank)

    for res in dense_results:
        fused_scores[res.doc_id] += dense_weight / (k_rrf + res.rank)

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


class HybridRRFRetriever:
    """Hybrid rank-fusion retriever combining BM25 lexical and dense search.

    Parameters
    ----------
    k_rrf : float, default=60.0
        Smoothing constant used in Reciprocal Rank Fusion.
    lexical_weight : float, default=1.0
        Relative weight for BM25 ranking scores.
    dense_weight : float, default=1.0
        Relative weight for dense ranking scores.
    bm25 : BM25Retriever, optional
        Lexical backend; defaults to a fresh BM25Retriever.
    dense : Retriever, optional
        Dense backend exposing ``index``/``search``; defaults to HashedTfidfRetriever.
    """

    def __init__(
        self,
        k_rrf: float = 60.0,
        lexical_weight: float = 1.0,
        dense_weight: float = 1.0,
        *,
        bm25: BM25Retriever | None = None,
        dense: Retriever | None = None,
    ) -> None:
        self.k_rrf = k_rrf
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight
        self.bm25 = bm25 if bm25 is not None else BM25Retriever()
        self.dense = dense if dense is not None else HashedTfidfRetriever()
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
        candidate_depth: int = 100,
    ) -> list[SearchResult]:
        """Perform hybrid reciprocal rank fusion retrieval."""
        if k <= 0:
            raise ValueError("Result count must be positive.")
        if not self.is_indexed:
            raise RuntimeError("Hybrid retriever is not indexed. Call 'index' first.")

        candidate_depth = max(candidate_depth, k)
        bm25_results = self.bm25.search(query, k=candidate_depth)
        dense_results = self.dense.search(query, k=candidate_depth)

        return reciprocal_rank_fusion(
            bm25_results,
            dense_results,
            k_rrf=self.k_rrf,
            lexical_weight=self.lexical_weight,
            dense_weight=self.dense_weight,
            k=k,
        )
