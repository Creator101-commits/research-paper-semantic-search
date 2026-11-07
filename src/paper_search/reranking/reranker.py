"""Lexical overlap reranking; the legacy class name is retained for compatibility."""

from __future__ import annotations

import time

from paper_search.data.schema import Document, SearchResult, normalize_text


class CrossEncoderReranker:
    """Second-stage lexical overlap reranker over top candidate subsets.

    Parameters
    ----------
    title_weight : float, default=2.0
        Importance multiplier for terms matching in the paper title.
    """

    def __init__(self, title_weight: float = 2.0) -> None:
        self.title_weight = title_weight
        self.doc_map: dict[str, Document] = {}

    def index_documents(self, documents: list[Document]) -> None:
        """Register corpus documents for cross-interaction text lookup."""
        self.doc_map = {doc.doc_id: doc for doc in documents}

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int = 10,
        max_candidate_depth: int = 50,
    ) -> tuple[list[SearchResult], float]:
        """Score candidate pairs with title and abstract token overlap and return reranked list."""
        t0 = time.perf_counter()
        query_tokens = set(normalize_text(query))
        if not query_tokens:
            return candidates[:top_k], 0.0

        pruned_candidates = candidates[:max_candidate_depth]
        reranked_scores: list[tuple[str, float]] = []

        for cand in pruned_candidates:
            if cand.doc_id not in self.doc_map:
                continue

            doc = self.doc_map[cand.doc_id]
            title_tokens = normalize_text(doc.title)
            abstract_tokens = normalize_text(doc.abstract)

            # Fine-grained interaction: title overlap + abstract overlap + first-stage prior
            title_overlap = sum(1 for t in title_tokens if t in query_tokens)
            abstract_overlap = sum(1 for t in abstract_tokens if t in query_tokens)

            # Interaction score
            interaction_score = (
                self.title_weight * title_overlap + abstract_overlap + 0.1 * cand.score
            )
            reranked_scores.append((cand.doc_id, interaction_score))

        # Sort descending by interaction score, deterministic tie-breaking
        sorted_candidates = sorted(
            reranked_scores,
            key=lambda item: (-item[1], item[0]),
        )[:top_k]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        results = [
            SearchResult(
                doc_id=doc_id,
                score=float(score),
                rank=r + 1,
            )
            for r, (doc_id, score) in enumerate(sorted_candidates)
        ]
        return results, elapsed_ms
